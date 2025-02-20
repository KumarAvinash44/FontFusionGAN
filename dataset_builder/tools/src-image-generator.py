import argparse
import glob
import io
import os

from PIL import Image, ImageFont, ImageDraw

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# Default data paths.
# Default data paths.
DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH,
                                  '../labels/2350-common-hangul.txt')
DEFAULT_FONTS_DIR = os.path.join(SCRIPT_PATH, '../src_font')
DEFAULT_TARGET_FONTS_DIR = os.path.join(SCRIPT_PATH, '../tgt_font')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_PATH, '../test')

# Width and height of the resulting image.
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128

def generate_hangul_images(label_file, fonts_dir, output_dir):
    """Generate Hangul image files.

    This will take in the passed in labels file and will generate several
    images using the font files provided in the font directory. The font
    directory is expected to be populated with *.ttf (True Type Font) files.
    The generated images will be stored in the given output directory. Image
    paths will have their corresponding labels listed in a CSV file.
    """
    with io.open(label_file, 'r', encoding='utf-8') as f:
        labels = f.read().splitlines()

    image_dir = os.path.join(output_dir, 'FMGAN-256-eval-src-font-images')
    if not os.path.exists(image_dir):
        os.makedirs(os.path.join(image_dir))

    # Get a list of the fonts.
    fonts = sorted(glob.glob(os.path.join(fonts_dir, '*.ttf')))

    # Get a list of the fonts.
    tgt_fonts = sorted(glob.glob(os.path.join(DEFAULT_TARGET_FONTS_DIR, '*.ttf')))

    total_count = 0
    prev_count = 0
    font_count = 0
    char_no = 0
    # Total number of font files is 
    print('total number of fonts are ', len(fonts))

    for character in labels:
        char_no += 1
        # Print image count roughly every 5000 images.
        if total_count - prev_count > 5000:
            prev_count = total_count
            print('{} images generated...'.format(total_count))
        for x in range(len(tgt_fonts)):
            font_count += 1
            
            for font in fonts:
                total_count += 1
                image = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT), color=255)
                font = ImageFont.truetype(font, 64)
                drawing = ImageDraw.Draw(image)
                w, h = drawing.textsize(character, font=font)
                drawing.text(
                    ((IMAGE_WIDTH-w)/2, (IMAGE_HEIGHT-h)/2),
                    character,
                    fill=0,
                    font=font
                )
                # file_string = '{}_{}.png'.format(font_count,char_no)
                file_string = '{}.png'.format(character)
                file_path = os.path.join(image_dir, file_string)
                image.save(file_path, 'PNG')
        font_count = 0
    char_no = 0

    print('Finished generating {} images.'.format(total_count))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label-file', type=str, dest='label_file',
                        default=DEFAULT_LABEL_FILE,
                        help='File containing newline delimited labels.')
    parser.add_argument('--font-dir', type=str, dest='fonts_dir',
                        default=DEFAULT_FONTS_DIR,
                        help='Directory of ttf fonts to use.')
    parser.add_argument('--target-font-dir', type=str, dest='target_fonts_dir',
                    default=DEFAULT_TARGET_FONTS_DIR,
                    help='Directory of ttf fonts to use.')
    parser.add_argument('--output-dir', type=str, dest='output_dir',
                        default=DEFAULT_OUTPUT_DIR,
                        help='Output directory to store generated images and '
                             'label CSV file.')
    args = parser.parse_args()
    generate_hangul_images(args.label_file, args.fonts_dir, args.output_dir)
