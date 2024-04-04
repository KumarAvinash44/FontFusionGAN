from PIL import Image
import os
import glob

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
img_path= os.path.join(SCRIPT_PATH, '../../results/2000(train_2000)/SW_GTH_index7')
imgs_path = glob.glob(os.path.join(img_path, '*.png'))

def get_name(path):
    name, _ = os.path.splitext(os.path.basename(path))
    return name

if all(get_name(path).isdigit() for path in imgs_path):
    src_imgs_path = sorted(imgs_path, key=lambda path: int(get_name(path)))
else:
    print("false")
    # src_imgs_path = sorted(src_imgs_path)
    src_imgs_path = sorted(imgs_path, key=lambda path: str(get_name(path)))

num = 0
for i in range(0, len(imgs_path), 10) :
    # output = Image.new('L', (128 * 2, 128 * 5))
    output = Image.new('L', (128 * 2 * 3, 128 * 5))

    for j in range(2):
        for k in range(5):
            im = Image.open(imgs_path[i + 5*j + k])
            # output.paste(im, (128 * j, 128 * k))
            output.paste(im, (128 * 3 * j, 128 * k))

    output.save(img_path + '/merge/merge' + str(num) + '.png')
    num += 1
