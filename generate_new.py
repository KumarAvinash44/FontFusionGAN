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
        src_img_path = os.path.join(test_path, "unseen_src_imgs")
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
            src_imgs_path = sorted(src_imgs_path, key=lambda path: int(get_name(path)))

        # Get a list of the ref characters.
        ref_img_path = os.path.join(test_path, "train_imgs")
        
        # for src in src_imgs_path:
        #     print(src)

        for i in range(no_fonts):
            for j in range(n_chars):
                rand = random.randint(1, no_fonts)
                ref_imgs_path = sorted(glob.glob(os.path.join(ref_img_path, str(i+1)+'_'+str(rand)+'*.png')))
                ref_chars.append(ref_imgs_path[-1])

        for i in range(0, len(ref_chars), n_chars):
            n_ref_chars.append(ref_chars[i:i+n_chars])

        for s, n in enumerate(n_ref_chars):
            ref_img = torch.stack([TRANSFORM(Image.open(r)) for r in n]).to(device)
            ref_batches = torch.split(ref_img, n_chars)
            font_name, _ = os.path.splitext(os.path.basename(fonts[s]))


            cl_feats = []
            for batch in ref_batches:
                _cl = encoder(batch)
                cl_feats.append(_cl[-1])
            cl_feats = torch.cat(cl_feats).mean(dim=0, keepdim=True)
            
            print("Processing font no. ", s+1)
            # preprocess src images
            for c, src_img in enumerate(src_imgs_path):
                src_img = TRANSFORM(Image.open(src_img)).to(device)
                cnt_feats = encoder(src_img.unsqueeze(0))
                sample, _ = g_ema([cnt_feats[-1], cl_feats], inject_index=5, input_is_latent=True)
                utils.save_image(
                    sample,
                    f"results/unseen_chars_seen_style/{font_name}_{c+1}.png",
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
        default="checkpoint/200000.pt",
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
