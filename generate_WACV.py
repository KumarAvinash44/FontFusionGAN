import json
import argparse
from pathlib import Path
from itertools import chain
from sconf import Config
from PIL import Image
import random
from os.path import join

import torch
from torchvision import transforms, utils

from model import Generator, Encoder

from base.dataset import sample
from base.utils import save_tensor_to_image, load_reference

import os, glob

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
fonts_dir= os.path.join(SCRIPT_PATH, 'dataset_builder/val_fonts')
no_fonts = len(glob.glob(os.path.join(fonts_dir, '*.ttf')))

TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def infer_FM_GAN(args, device, mean_latent, size, gen, encoder, save_dir, source_path, source_ext, gen_chars, key_ref_dict, load_img, batch_size=1, return_img=False):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if source_ext == "ttf":
        source = read_font(source_path)
        gen_chars = get_filtered_chars(source) if gen_chars is None else gen_chars

        def read_source(char):
            return render(source, char)
    else:
        source = Path(source_path)
        gen_chars = [p.stem for p in source.glob(f"*.{source_ext}")] if gen_chars is None else gen_chars

        def read_source(char):
            impath = source / f"{char}.png"
            return Image.open(str(impath))

    key_gen_dict = {k: gen_chars for k in key_ref_dict}
    outs = {}

    for key, gchars in key_gen_dict.items():
        (save_dir / key).mkdir(parents=True, exist_ok=True)


        ref_chars = key_ref_dict[key]
        print(key)
        print()
        ref_imgs = torch.stack([TRANSFORM(load_img(key, c)) for c in ref_chars]).cuda(3)
        ref_batches = torch.split(ref_imgs, batch_size)

        cl_feats = []
        for batch in ref_batches:
            _cl = encoder(batch)
            cl_feats.append(_cl)
        cl_feats = torch.cat(cl_feats[-1]).mean(dim=0, keepdim=True)

        for char in gchars:
            source_img = TRANSFORM(read_source(char)).unsqueeze(0).cuda(3)

            _co = encoder(source_img)
            out = gen([_co[-1], cl_feats], inject_index=5, input_is_latent=True)[0].detach().cpu()

            if return_img:
                outs.setdefault(key, []).append(out)

            path = save_dir / key / f"{char}.png"
            # save_tensor_to_image(out, path)
            # custom pytorch save
            utils.save_image(
                out,
                path,
                normalize=True,
                range=(-1, 1),
            )

    return outs

def main():
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    print("Device chosen is ", device)
    print()

    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", help="path to save the result file")
    parser.add_argument("--seed", type=int, default=1304, help="path to save the result file")
    parser.add_argument("--size", type=int, default=128, help="output image size of the generator")
    parser.add_argument("--sample", type=int, default=1, help="number of samples to be generated for each image")
    parser.add_argument("--pics", type=int, default=30, help="number of images to be generated")
    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument("--truncation_mean", type=int, default=4096, help="number of vectors to calculate mean for the truncation",)
    parser.add_argument("--ckpt", type=str, default="checkpoint/310000.pt", help="path to the model checkpoint")
    parser.add_argument("--channel_multiplier", type=int, default=1, help="channel multiplier of the generator. config-f = 2, else = 1")
    parser.add_argument('--test_path', type=str, default='dataset', help="path to the test dataset")
    parser.add_argument("--n_ref", type=int, default=1, help="number of reference characters to use")
    

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

    # generate(args, g_ema, encoder, device, mean_latent, args.test_path, args.size)

    # args, left_argv = parser.parse_known_args()
    # args, cfg, gen_model, infer_func, infer_args = setup_eval_config(args, left_argv)
    # gen = load_model(args, cfg, gen_model)

    random.seed(args.seed)

    data_dir = join(args.test_path, "unseen_ref_imgs")
    source_path = join(args.test_path, "unseen_src_imgs")
    extension = "png"
    source_ext="png"
    ref_chars = "害树案排"
    # ref_chars = None

    key_ref_dict, load_img = load_reference(data_dir, extension, ref_chars)

    infer_FM_GAN(args, device, mean_latent, args.size, gen=g_ema, encoder=encoder,
               save_dir=args.result_dir,
               source_path=source_path,
               source_ext=source_ext,
               gen_chars=None,
               key_ref_dict=key_ref_dict,
               load_img=load_img,
               )


if __name__ == "__main__":
    main()