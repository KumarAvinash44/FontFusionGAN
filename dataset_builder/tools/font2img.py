from PIL import Image,ImageDraw,ImageFont
import matplotlib.pyplot as plt
import os
import numpy as np
import pathlib
import argparse
import io
import numpy
import glob



SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# Default data paths.
DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH,
                               '../labels/2350-common-hangul.txt')
# DEFAULT_FONTS_DIR = os.path.join(SCRIPT_PATH, '../train_fonts')
DEFAULT_FONTS_DIR = os.path.join(SCRIPT_PATH, '../tgt_font')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_PATH, 'test')


parser = argparse.ArgumentParser(description='Obtaining characters from .ttf')
parser.add_argument('--ttf_path', type=str, default=DEFAULT_FONTS_DIR, help='ttf directory')
parser.add_argument('--charset', type=str, default=DEFAULT_LABEL_FILE , help='characters')
parser.add_argument('--save_path', type=str, default=DEFAULT_OUTPUT_DIR, help='images directory')
parser.add_argument('--img_size', type=int,  default='128', help='The size of generated images')
parser.add_argument('--chara_size', type=int, default='100', help='The size of generated characters')
args = parser.parse_args()

# file_object = open(args.charset,encoding='utf-8')  
with io.open(args.charset, 'r', encoding='utf-8-sig') as f:
    characters = f.read().splitlines()

# try:
#   characters = file_object.read()
# # finally:
# #     # file_object.close()


def draw_single_char(ch, font, canvas_size, x_offset, y_offset):
    img = Image.new("L", (canvas_size, canvas_size), color=255)
    draw = ImageDraw.Draw(img)
    draw.text((x_offset, y_offset), ch, 0, font=font)
    return img

def draw_example(ch, src_font, canvas_size, x_offset, y_offset):
    src_img = draw_single_char(ch, src_font, canvas_size, x_offset, y_offset)
    example_img = Image.new("L", (canvas_size, canvas_size), color=255)
    example_img.paste(src_img, (0, 0))
    return example_img

def is_not_existing(image, character):
    """
    Check the image whether this is a blank image or not
     :param image: numpy.ndarray, image array
    :param character: char, character
    :return: boolean
    """

    if character == " ": 
        return False
    else:
        if 0 in image:
            return False
        return True


data_dir = args.ttf_path
data_root = pathlib.Path(data_dir)
print(data_root)

all_image_paths = list(sorted(glob.glob(os.path.join(data_root, '*.ttf'))))
all_image_paths = [str(path) for path in all_image_paths]
print("Total no of fonts are ", len(all_image_paths))
print()
for i in range (len(all_image_paths)):
    print(os.path.basename(os.path.normpath(all_image_paths[i])))

seq = list()

# You need to edit
image_dir = os.path.join(args.save_path, 'Name_prn')
os.makedirs(os.path.join(image_dir))

for (label,item) in zip(range(len(all_image_paths)),all_image_paths):
    src_font = ImageFont.truetype(item, size = args.chara_size)
    for (chara,cnt) in zip(characters, range(len(characters))):
        img = draw_example(chara, src_font, args.img_size, (args.img_size-args.chara_size)/2, (args.img_size-args.chara_size)/2)
        font_name = os.path.basename(os.path.normpath(item))
        path_full = os.path.join(image_dir, font_name)
        if not os.path.exists(path_full):
            os.mkdir(path_full)
        image_arr = numpy.array(img)
        if is_not_existing(image_arr, chara):
            continue
        img.save(os.path.join(path_full, "%s.png" % str(chara)))
        