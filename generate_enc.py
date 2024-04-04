import argparse
from torchvision import transforms, utils
import os
import glob
from PIL import Image
import random


import torch
from torchvision import utils
from model import Generator, Encoder
from tqdm import tqdm
from torch.utils import data
from custom_dataset_loader_test import DatasetFromFolder

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
fonts_dir= os.path.join(SCRIPT_PATH, 'dataset_builder/train_fonts')
no_fonts = len(glob.glob(os.path.join(fonts_dir, '*.ttf')))

TRANSFORM = transforms.Compose([transforms.ToTensor(),
                          transforms.Normalize([0.5], [0.5])])

def get_name(path):
    name, _ = os.path.splitext(os.path.basename(path))
    return name

def generate(args, g_ema, encoder, device, mean_latent, test_path, size):
    with torch.no_grad():
        g_ema.eval()
        encoder.eval()
        ref_chars = []
        n_chars = 1
        n_ref_chars = []
        cl_feats = []
        # Get a list of the src characters.
        # You need to edit
        # src_img_path = os.path.join(test_path, "test_img_2000/handwritten/UhBee dami.ttf")
        src_img_path = os.path.join(test_path, "test_img_2000/handwritten/UhBee dami.ttf")
        src_imgs_path = glob.glob(os.path.join(src_img_path, '*.png'))
        fonts = sorted(glob.glob(os.path.join(fonts_dir, '*.ttf')))

        # if the image names are numbers, sort by the value rather than asciibetically
        # having sorted inputs means that the outputs are sorted in test mode
        if all(get_name(path).isdigit() for path in src_imgs_path):
            print("true")
            src_imgs_path = sorted(src_imgs_path, key=lambda path: int(get_name(path)))
        else:
            print("false")
            # src_imgs_path = sorted(src_imgs_path)
            src_imgs_path = sorted(src_imgs_path, key=lambda path: str(get_name(path)))

        # Get a list of the ref characters.
        # You need to edit
        ref_img_path = os.path.join(test_path, "test_img_2000/printed/DXMyeongjo_20.ttf")
        # ref_img_path = os.path.join(test_path, "test_img_2000/printed/DXGothic_20.ttf")
        
        # for src in src_imgs_path:
        #     print(src)

        ref_imgs_path = sorted(glob.glob(os.path.join(ref_img_path, '*.png')))
        n_chars = len(ref_imgs_path)

        for i in range(n_chars):
            ref_chars.append(ref_imgs_path[i][-5])

        for i in range(len(ref_chars)):
            # preprocess ref images
            ref_img = torch.stack([TRANSFORM(Image.open(ref_imgs_path[i]))]).to(device)
            ref_batches = torch.split(ref_img, n_chars)

            cl_feats = []
            for batch in ref_batches:
                _cl = encoder(batch)
                cl_feats.append(_cl[-1])
            cl_feats = torch.cat(cl_feats).mean(dim=0, keepdim=True)
                
            # preprocess src images
            src_img = TRANSFORM(Image.open(src_imgs_path[i])).to(device)
            src_img = src_img.unsqueeze(0)
            cnt_feats = encoder(src_img)

            # You need to edit
            # sample, _ = g_ema([cnt_feats[-1]], inject_index=7, input_is_latent=True)
            sample, _ = g_ema([cnt_feats[-1], cl_feats], inject_index=6, input_is_latent=True)
            # sample = torch.cat((src_img, ref_img, sample), dim =- 1)
            utils.save_image(
                src_img,
                # You need to edit
                # f"results/avinash_generated/index6/{ref_chars[i]}.png",
                f"results/avinash_generated/myeong_mixed/{ref_chars[i]}_src.png",
                nrow=5,
                normalize=True,
                range=(-1, 1),
            )            
            
            utils.save_image(
                ref_img,
                # You need to edit
                # f"results/avinash_generated/index6/{ref_chars[i]}.png",
                f"results/avinash_generated/myeong_mixed/{ref_chars[i]}_ref.png",
                nrow=5,
                normalize=True,
                range=(-1, 1),
            )
            utils.save_image(
                sample,
                # You need to edit
                # f"results/avinash_generated/index6/{ref_chars[i]}.png",
                f"results/avinash_generated/myeong_mixed/{ref_chars[i]}_mix.png",
                nrow=5,
                normalize=True,
                range=(-1, 1),
            )
            if i % 500 == 0:
                print("processed images => ", i)

    

       # # preprocess ref images
       #  for s, ref_style in enumerate(ref_chars):
       #      ref_img = TRANSFORM(Image.open(ref_style).convert('RGB')).to(device)
       #      style_feats = encoder(ref_img.unsqueeze(0))
       #      # preprocess src images
       #      for c, src_img in enumerate(src_imgs_path):
       #          src_img = TRANSFORM(Image.open(src_img).convert('RGB')).to(device)
       #          cnt_feats = encoder(src_img.unsqueeze(0))
       #          sample, _ = g_ema([cnt_feats[-1], style_feats[-1]], inject_index=5, input_is_latent=True)
       #          utils.save_image(
       #              sample,
       #              f"sample/results/generated_unseen/{s+1}_{c+1}.png",
       #              normalize=True,
       #              range=(-1, 1),
       #          )
       #          # utils.save_image(
       #          #     tgt_img,
       #          #     f"sample/results/target/{img_name[0]}.png",
       #          #     normalize=True,
       #          #     range=(-1, 1),
       #          # )
       #          if i % 500 == 0:
       #              print("processed images => ", i)              

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
        "--pics", type=int, default=30, help="number of images to be generated"
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
        # You need to edit
        default="checkpoint/korean_2350.pt",
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
    checkpoint = torch.load(args.ckpt, map_location="cuda:0")

    g_ema.load_state_dict(checkpoint["g_ema"])
    encoder.load_state_dict(checkpoint["enc"])

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    generate(args, g_ema, encoder, device, mean_latent, args.test_path, args.size)