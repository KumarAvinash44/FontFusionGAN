import argparse
import math
import random
import os

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils, datasets
from tqdm import tqdm

from torchvision.datasets import ImageFolder

try:
    import wandb

except ImportError:
    wandb = None


# from dataset import MultiResolutionDataset
from custom_dataset_loader import DatasetFromFolder
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from op import conv2d_gradfix
from non_leaking import augment, AdaptiveAugment


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty

mse_loss = nn.MSELoss(size_average=True)

def recon_criterion(predict, target):
    return torch.mean(torch.abs(predict - target))

def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None

def train(args, loader, generator, discriminator, encoder, g_optim, d_optim, e_optim, g_ema, device):
    loader = sample_data(loader)

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    g_w_rec_loss_val = 0
    g_x_rec_loss_val = 0
    enc_x_rec_loss_val = 0
    enc_fm_loss_val = 0
    enc_w_rec_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module
        e_module = encoder.module

    else:
        g_module = generator
        d_module = discriminator
        e_module = encoder

    accum = 0.5 ** (32 / (10 * 1000))
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    r_t_stat = 0

    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, 8, device)

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")

            break

        real_img, _ = next(loader)
        real_img = real_img.to(device) 

        requires_grad(generator, False)
        requires_grad(encoder, False)
        requires_grad(discriminator, True)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, _ = generator(noise)

        if args.augment:
            real_img_aug, _ = augment(real_img, ada_aug_p)
            fake_img, _ = augment(fake_img, ada_aug_p)

        else:
            real_img_aug = real_img

        fake_pred, _ = discriminator(fake_img)
        real_pred, _ = discriminator(real_img_aug)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        if args.augment and args.augment_p == 0:
            ada_aug_p = ada_augment.tune(real_pred)
            r_t_stat = ada_augment.r_t_stat

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True

            if args.augment:
                real_img_aug, _ = augment(real_img, ada_aug_p)

            else:
                real_img_aug = real_img

            real_pred, _ = discriminator(real_img_aug)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict["r1"] = r1_loss

        requires_grad(generator, True)
        requires_grad(encoder, True)
        requires_grad(discriminator, False)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, stylecode = generator(noise, return_latents=True)

        if args.augment:
            fake_img, _ = augment(fake_img, ada_aug_p)

        fake_pred, _ = discriminator(fake_img)
        adv_loss = g_nonsaturating_loss(fake_pred)

        fake_stylecode = encoder(fake_img)   #important
        g_w_rec_loss = mse_loss(stylecode, fake_stylecode[-1])  #important

        # fake_img_2, _ = generator(fake_stylecode, input_is_latent=True)
        # g_x_rec_loss = recon_criterion(fake_img_2, fake_img)

        # g_loss = adv_loss + g_w_rec_loss * args.lambda_w_rec_loss + g_x_rec_loss * args.lambda_x_rec_loss

        g_loss = adv_loss


        loss_dict["g"] = adv_loss
        loss_dict["g_w_rec_loss"] = g_w_rec_loss
        # loss_dict["g_x_rec_loss"] = g_x_rec_loss

        generator.zero_grad()
        g_loss.backward(retain_graph=True)
        g_optim.step()
        
        encoder.zero_grad()
        (g_w_rec_loss * args.lambda_w_rec_loss).backward()
        e_optim.step()

        accumulate(g_ema, g_module, accum)

        # Train Encoder
        requires_grad(generator, True)
        requires_grad(encoder, True)
        requires_grad(discriminator, False)

        fake_stylecode = encoder(real_img)
        fake_img, _ = generator(fake_stylecode, input_is_latent=True)
        enc_x_rec_loss = recon_criterion(fake_img, real_img)

        fake_stylecode_2 = encoder(fake_img)
        enc_w_rec_loss = mse_loss(fake_stylecode[-1], fake_stylecode_2[-1])

        _, real_feats = discriminator(real_img)
        _, fake_feats = discriminator(fake_img)
        fm_loss = recon_criterion(fake_feats.mean(3).mean(2), real_feats.mean(3).mean(2))

        encoder_loss = enc_x_rec_loss * args.lambda_x_rec_loss + fm_loss * args.lambda_perceptual_loss + enc_w_rec_loss * args.lambda_w_rec_loss 
        loss_dict["enc_x_rec_loss"] = enc_x_rec_loss
        loss_dict["enc_fm_loss"] = fm_loss
        loss_dict["enc_w_rec_loss"] = enc_w_rec_loss
        
        encoder.zero_grad()
        generator.zero_grad()
        encoder_loss.backward()        
        e_optim.step()
        g_optim.step()

        loss_reduced = reduce_loss_dict(loss_dict)

        # D losses
        d_loss_val = loss_reduced["d"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        # predictions
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        # G losses
        g_loss_val = loss_reduced["g"].mean().item()
        g_w_rec_loss_val = loss_reduced["g_w_rec_loss"].mean().item()
        # g_x_rec_loss_val = loss_reduced["g_x_rec_loss"].mean().item()
        # E losses
        enc_x_rec_loss_val = loss_reduced["enc_x_rec_loss"].mean().item()
        enc_fm_loss_val = loss_reduced["enc_fm_loss"].mean().item()
        enc_w_rec_loss_val = loss_reduced["enc_w_rec_loss"].mean().item()
        

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f};"
                    f"g_W_rec_loss: {g_w_rec_loss_val:.4f}; "
                    f"enc_fm_loss: {enc_fm_loss_val:.4f}; enc_x_rec_loss: {enc_x_rec_loss_val:.4f}; enc_w_rec_loss: {enc_w_rec_loss_val:.4f};"
                )
            )

            if wandb and args.wandb:
                wandb.log(
                    {
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
                        "Augment": ada_aug_p,
                        "Rt": r_t_stat,
                        "R1": r1_val,
                        "Path Length Regularization": path_loss_val,
                        "Mean Path Length": mean_path_length,
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,
                        "Path Length": path_length_val,
                    }
                )

            if i % 10000 == 0:
                with torch.no_grad():
                    g_ema.eval()
                    encoder.eval()
                    # loading data
                    img, _ = next(loader)
                    img = img.to(device)
                    img2, _ = next(loader)
                    img2 = img2.to(device)

                    # Calling model
                    feats = encoder(img)
                    feats2 = encoder(img2)
                    sample, _ = g_ema([feats[-1], feats2[-1]], inject_index=5, input_is_latent=True)
                    
                    sample = torch.cat((img, img2, sample), dim=-1)
                    utils.save_image(
                        sample,
                        # You need to edit
                        f"sample_new/{str(i).zfill(6)}.png",
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )

            if i % 10000 == 0:
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "enc": e_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(), 
                        "e_optim": e_optim.state_dict(),
                        "args": args,
                        "ada_aug_p": ada_aug_p,
                    },
                    # You need to edit
                    f"checkpoint_new/{str(i).zfill(6)}.pt",
                )


if __name__ == "__main__":
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    print("Device chosen is ", device)
    print()

    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")

    parser.add_argument('--path', type=str, default='dataset/train_img_2350', help="path to the dataset")
    parser.add_argument('--arch', type=str, default='stylegan2', help='model architectures (stylegan2 | swagan)')
    parser.add_argument(
        "--iter", type=int, default=400000, help="total training iterations"
    )
    parser.add_argument(
        "--batch", type=int, default=8, help="batch sizes for each gpus"
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=4,
        help="number of the samples generated during training",
    )
    parser.add_argument(
        "--size", type=int, default=128, help="image sizes for the model"
    )
    parser.add_argument(
        "--r1", type=float, default=10, help="weight of the r1 regularization"
    )
    parser.add_argument(
        "--path_regularize",
        type=float,
        default=2,
        help="weight of the path length regularization",
    )
    parser.add_argument(
        "--path_batch_shrink",
        type=int,
        default=2,
        help="batch size reducing factor for the path length regularization (reduce memory consumption)",
    )
    parser.add_argument(
        "--d_reg_every",
        type=int,
        default=16,
        help="interval of the applying r1 regularization",
    )
    parser.add_argument(
        "--g_reg_every",
        type=int,
        default=4,
        help="interval of the applying path length regularization",
    )
    parser.add_argument(
        "--mixing", type=float, default=0.9, help="probability of latent code mixing"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        # default="checkpoint/740000.pt",
        help="path to the checkpoints to resume training",
    )
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=1,
        help="channel multiplier factor for the model. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="use weights and biases logging"
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )
    parser.add_argument(
        "--augment", action="store_true", help="apply non leaking augmentation"
    )
    parser.add_argument(
        "--augment_p",
        type=float,
        default=0,
        help="probability of applying augmentation. 0 = use adaptive augmentation",
    )
    parser.add_argument(
        "--ada_target",
        type=float,
        default=0.6,
        help="target augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_length",
        type=int,
        default=500 * 1000,
        help="target duraing to reach augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_every",
        type=int,
        default=256,
        help="probability update interval of the adaptive augmentation",
    )
    parser.add_argument("--lambda_w_rec_loss", type=float, default=1)
    parser.add_argument("--lambda_x_rec_loss", type=float, default=1)
    parser.add_argument("--lambda_perceptual_loss", type=float, default=1)

    args = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.latent = 64
    args.n_mlp = 8

    args.start_iter = 0

    if args.arch == 'stylegan2':
        from model import Generator, Discriminator, Encoder

    elif args.arch == 'swagan':
        from swagan import Generator, Discriminator

    generator = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)
    encoder = Encoder(
        args.size, args.latent, channel_multiplier=args.channel_multiplier
    ).to(device)

    # g_reg_ratio = 1 = args.g_reg_every / (args.g_reg_every + 1)
    g_reg_ratio = 1
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr,
        betas=(0, 0.99),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )
    e_optim = optim.Adam(
        encoder.parameters(),
        lr=args.lr,
        betas=(0, 0.99),
    )

    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        generator.load_state_dict(ckpt["g"])
        discriminator.load_state_dict(ckpt["d"])
        encoder.load_state_dict(ckpt["enc"])
        g_ema.load_state_dict(ckpt["g_ema"])         

        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"]) 
        e_optim.load_state_dict(ckpt["e_optim"])

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        encoder = nn.parallel.DistributedDataParallel(
            encoder,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]
    )

    dataset = ImageFolder(args.path, transform)
    # dataset = DatasetFromFolder(args.path, args.size)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project="stylegan 2")

    train(args, loader, generator, discriminator, encoder, g_optim, d_optim, e_optim, g_ema, device)
