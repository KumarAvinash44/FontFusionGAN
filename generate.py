import argparse

import torch
import os, glob
from torchvision import utils
from model import Generator, Encoder
from tqdm import tqdm
from torch.utils import data
from custom_dataset_loader_test import DatasetFromFolder
from torchvision import transforms, utils
from PIL import Image
import random


TRANSFORM = transforms.Compose([transforms.ToTensor(),
                          transforms.Normalize([0.5], [0.5])])


def data_sampler(dataset, shuffle, distributed=False):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def get_name(path):
    name, _ = os.path.splitext(os.path.basename(path))
    return name


def generate_src_random(args, g_ema, encoder, device, mean_latent, test_path, size):
    with torch.no_grad():
        g_ema.eval()
        encoder.eval()
        # Get a list of the src characters.
        src_img_path = os.path.join(test_path, "seen_src_imgs")
        src_imgs_path = glob.glob(os.path.join(src_img_path, '*.png'))
        # if the image names are numbers, sort by the value rather than asciibetically
        # having sorted inputs means that the outputs are sorted in test mode
        # if all(get_name(path).isdigit() for path in src_imgs_path):
        #     print("true")
        #     src_imgs_path = sorted(src_imgs_path, key=lambda path: int(get_name(path)))
        # else:
        #     print("false")
        #     # src_imgs_path = sorted(src_imgs_path)
        #     src_imgs_path = sorted(src_imgs_path, key=lambda path: int(get_name(path)))


        sample_z1 = torch.randn(args.sample, args.latent, device=device)
        # preprocess src images
        for c, src_img in enumerate(src_imgs_path):
            src_img = TRANSFORM(Image.open(src_img)).to(device)
            cnt_feats = encoder(src_img.unsqueeze(0))
            # sample_z2 = torch.randn(args.sample, args.latent, device=device)
            sample, _ = g_ema(
                [cnt_feats[-1], sample_z1], inject_index=3)

            utils.save_image(
                sample,
                f"results/latent_guided/src_random_style_avinash/exp1/{c+1}.png",
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )

def generate_random(args, g_ema, device, mean_latent):

    with torch.no_grad():
        g_ema.eval()
        sample_z1 = torch.randn(args.sample, args.latent, device=device)
        for i in tqdm(range(args.pics)):
            sample_z2 = torch.randn(args.sample, args.latent, device=device)
            sample, _ = g_ema(
                [sample_z1, sample_z2], inject_index=5)

            utils.save_image(
                sample,
                f"results/latent_guided/random/style_9/{str(i).zfill(6)}.png",
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )


def generate(args, g_ema, encoder, device, mean_latent, test_path, size):

    with torch.no_grad():
        g_ema.eval()
        encoder.eval()
        # Loading test dataset
        dataset = DatasetFromFolder(test_path, size)
        test_loader = data.DataLoader(
            dataset,
            batch_size=1,
            sampler=data_sampler(dataset, shuffle=False)) 

        test_loader = sample_data(test_loader)
        print("total number of images to be processed", len(dataset))
        for i in range(len(dataset)):
            cnt_img, style_img, tgt_img, img_name = next(iter(test_loader))
            #loading data
            cnt_img = cnt_img.to(device)
            style_img = style_img.to(device)
            tgt_img = tgt_img.to(device)
            # Calling model
            cnt_feats = encoder(cnt_img)
            style_feats = encoder(style_img)
            sample, _ = g_ema([cnt_feats[-1], style_feats[-1]], inject_index=5, input_is_latent=True)
            sample = torch.cat((cnt_img, style_img, sample), dim = -1)
            # sample = torch.cat((cnt_img, style_img, sample, tgt_img), dim=-1)
            utils.save_image(
                sample,
                f"results/avinash_generated/font1_update/{img_name[0]}.png",
                normalize=True,
                range=(-1, 1),
            )
            # utils.save_image(
            #     tgt_img,
            #     f"sample/results/target/{img_name[0]}.png",
            #     normalize=True,
            #     range=(-1, 1),
            # )
            if i % 500 == 0:
                print("processed images => ", i)


if __name__ == "__main__":
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print("Device chosen is ", device)
    print()

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--size", type=int, default=128, help="output image size of the generator"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="number of samples to be generated for each image",
    )
    parser.add_argument(
        "--pics", type=int, default=100, help="number of images to be generated"
    )
    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpointTest/001600.pt",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=1,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )
    parser.add_argument('--test_path', type=str, default='dataset',
         help="path to the test dataset")
    
    args = parser.parse_args()

    args.latent = 64
    args.n_mlp = 8

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    encoder = Encoder(
        args.size, args.latent, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.ckpt)

    g_ema.load_state_dict(checkpoint["g_ema"])
    encoder.load_state_dict(checkpoint["enc"])

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    generate(args, g_ema, encoder, device, mean_latent, args.test_path, args.size)
    # generate_random(args, g_ema, device, mean_latent)    #It is working
    # generate_src_random(args, g_ema, encoder, device, mean_latent, args.test_path, args.size)    #it is working
