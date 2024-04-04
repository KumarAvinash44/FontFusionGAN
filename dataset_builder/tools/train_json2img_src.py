import argparse
import glob
import io
import os
import numpy
import json

from PIL import Image, ImageFont, ImageDraw
from fontTools.ttLib import TTFont

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# Default data paths.
DEFAULT_FONTS_DIR = os.path.join(SCRIPT_PATH, '../train_fonts')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_PATH, '../../dataset2')
UNICODE_CHAR_DIR = os.path.join(SCRIPT_PATH, "./charset/en_all.json")

# Width and height of the resulting image.
# 결과 이미지의 높이 너비
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128

total_count = 0

def char_in_font(unicode_char, font):
    """
    Check whether the character is existing in font
    :param unicode_char: int, unicode of a character you want check
    :param font: TTFont class, font style
    :return: boolean
    """
    for y in font["cmap"].tables[0].cmap.keys():
        if unicode_char in chr(y):
            return True

    return False

def get_char_list(font):
    """
    :return lang_unicode: string list, cll characters
    """
    raw_list = json.load(open(UNICODE_CHAR_DIR))
    char_list = []

    for character in raw_list:
        if char_in_font(character, font):
            char_list.append(character)

    return char_list


def draw_images(fonts, image_dir):
    """
    Draw character images styling each of fonts
    :param fonts: string list
    :param image_dir: string
    :param labels_csv: csv object
    :param char_list: list of characters
    """
    global total_count
    prev_count = 0

    # counter = sorted(glob.glob(os.path.join(DEFAULT_SRC_FONTS_DIR, '*.ttf')))
    # print(len(counter))

    # for c, count in enumerate(counter):
    for font_index, fontpath in enumerate(fonts):
        font = TTFont(fontpath)
        # Get characters of this font
        char_list = get_char_list(font)

        # Get Font name for saving image with font name
        font_path = fontpath
        _, tail = os.path.split(font_path)
        font_name = tail.split('.')[0]

        print('Font no. {}, named {} has {} characters.'.format(font_index, font_name, len(char_list)))
        print()
        for cc, character in enumerate(char_list):

            if total_count - prev_count > 5000:
                prev_count = total_count
                print('{} images generated...'.format(total_count))

            image = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT), color="white")
            font = ImageFont.truetype(fontpath, 100)
            drawing = ImageDraw.Draw(image)
            w, h = drawing.textsize(character, font=font)
            drawing.text(
                ((IMAGE_WIDTH - w) / 2, (IMAGE_HEIGHT - h) / 2),
                character,
                fill=(0),
                font=font
            )

            image_arr = numpy.array(image)
            if is_not_existing(image_arr, character):
                continue

            total_count += 1
            file_string = '{}.png'.format(cc+1)
            file_path = os.path.join(image_dir, file_string)
            image.save(file_path, 'PNG')


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


def generate_fonts_images(fonts_dir, output_dir):
    # Get a list of the fonts.
    fonts = sorted(glob.glob(os.path.join(fonts_dir, '*.ttf')))

    # Create font folders with name
    image_dir = os.path.join(output_dir, "val_unseen_src")
    if not os.path.exists(image_dir):
        os.makedirs(os.path.join(image_dir))

    draw_images(fonts, image_dir)


    print('Finished generating {} images.'.format(total_count))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--font-dir', type=str, dest='fonts_dir',
                        default=DEFAULT_FONTS_DIR,
                        help='Directory of ttf fonts to use.')
    parser.add_argument('--output-dir', type=str, dest='output_dir',
                        default=DEFAULT_OUTPUT_DIR,
                        help='Output directory to store generated images and '
                             'label CSV file.')
    args = parser.parse_args()

    generate_fonts_images(args.fonts_dir, args.output_dir)