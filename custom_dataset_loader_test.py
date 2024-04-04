from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, utils
from torch.utils import data
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
import os
import random
from os.path import join
from os import listdir
import glob

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
# Default paths.
Lable_file = os.path.join(SCRIPT_PATH,
                                  'dataset_builder/labels/350-common-hangul.txt')
font_path = os.path.join(SCRIPT_PATH, 'dataset_builder/val_fonts')

# total_characters = sum(1 for _ in open(Lable_file, encoding='utf-8'))
total_characters = 50
# total_styles = len(glob.glob1(font_path,"*.ttf"))
total_styles = 1

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def replace_name(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)

def gen_random_no(start, stop):
    return random.randint(start, stop)

def replace_style(string, new, I):
    li = string[string.index(I):]
    # print("new", new)
    # print("li", li)
    # print(new+li)
    return new+li

def get_positive_img(img_name, total_chars):
    file_name = img_name
    name, file_extension = os.path.splitext(file_name)
    img_name = name + file_extension
    style_no = name.split('_')[0]
    char_no = name.split('_')[1]
    # Get random character
    char = random.randint(1,total_chars)
    style_img_name = replace_name(str(img_name), str(char_no), str(char), 1)
    return style_img_name

def get_negative_img(a_img_name, total_styles, total_chars):
    a_file_name = a_img_name
    a_name, a_file_extension = os.path.splitext(a_file_name)
    a_img_name = a_name + a_file_extension
    a_style_no = a_name.split('_')[0]
    a_char_no = a_name.split('_')[1]

    # Get random style
    n_img_style = gen_random_no(1, total_styles)

    while True:

        if int(n_img_style) == int(a_style_no):
            n_img_style  = gen_random_no(1, total_styles)
        else:
            n_img_style = n_img_style
            break

    style_name = replace_style(str(a_img_name), str(n_img_style), I="_")
    
    # n_file_name = style_name
    # n_name, n_file_extension = os.path.splitext(n_file_name)
    # n_img_name = n_name + n_file_extension
    # n_style_no = n_name.split('_')[0]
    # n_char_no = n_name.split('_')[1]
    # # Get random character
    # char = random.randint(1, total_chars)
    # n_img = replace_name(str(n_img_name), str(n_char_no), str(char), 1)

    return style_name

def get_src_img(img_name, total_chars):
    file_name = img_name
    name, file_extension = os.path.splitext(file_name)
    img_name = name + file_extension
    src_style = 1
    src_img_name = replace_style(str(img_name), str(src_style), I="_")
    return src_img_name

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, img_size):
        super(DatasetFromFolder, self).__init__()        
        self.src_img_path = join(image_dir, "test_img_50/printed/font")
        self.b_path = join(image_dir, "fineTune/font1") 
        self.image_filenames = sorted([x for x in listdir(self.b_path) if is_image_file(x)])
        self.img_size = img_size
        self.total_chars = total_characters
        self.total_styles = total_styles
       
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize([0.5], [0.5])]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        # Get Images
        # a = Image.open(join(self.a_path, self.image_filenames[index])).convert('RGB')
        anchor_img = Image.open(join(self.b_path, self.image_filenames[index]))
        # print("anchor_img", self.image_filenames[index])
        
        # Get source image corresponding to the anchor character
        src_img_name = get_src_img(self.image_filenames[index], self.total_chars)
        negative_img = self.transform(Image.open(join(self.src_img_path, src_img_name)))        

        positive_img_name = get_positive_img(self.image_filenames[index], self.total_chars)
        positive_img = Image.open(join(self.b_path, positive_img_name))
        # print("positive_img", positive_img_name)

        # negative_img_name = get_negative_img(self.image_filenames[index], self.total_styles, self.total_chars)
        # negative_img = Image.open(join(self.b_path, negative_img_name)).convert('RGB')
        # # print("negative_img", negative_img_name)
        # # print()

        # # Get Labels
        # style_label = int((self.image_filenames[index].split('_'))[0].split('.')[0])
        # char_label = int((self.image_filenames[index].split('_'))[1].split('.')[0])
        # anchor_s_label = style_label - 1 # This is done to avoid the lable indexing label as our labels start with 1 and not 0
        # anchor_c_label = char_label - 1 # This is done to avoid the lable indexing label as our labels start with 1 and not 0

        # a = self.transform(a)
        anchor_img = self.transform(anchor_img)
        positive_img = self.transform(positive_img)
        # negative_img = self.transform(negative_img)

        file_name =  self.image_filenames[index]
        name, _ = os.path.splitext(file_name)
        return negative_img, positive_img, anchor_img, name

    def __len__(self):
        return len(self.image_filenames)

def plot_data(loader):
    # Plot some training images
    anchor_img, positive_img, negative_img, anchor_c_label, anchor_s_label = next(iter(loader))

    # plt.figure(figsize=(8,8))
    # plt.axis("off")
    # plt.title("src_img training Images")  
    # plt.imshow(np.transpose(vutils.make_grid(src_img[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    # plt.show()

    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("anchor_img training Images")      
    plt.imshow(np.transpose(vutils.make_grid(anchor_img[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()

    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("positive_img training Images")      
    plt.imshow(np.transpose(vutils.make_grid(positive_img[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()


    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("negative_img training Images")      
    plt.imshow(np.transpose(vutils.make_grid(negative_img[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()