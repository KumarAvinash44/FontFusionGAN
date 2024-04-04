from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, utils
from torch.utils import data

import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
from os.path import join
from os import listdir

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, img_size):
        super(DatasetFromFolder, self).__init__()        
        # self.src_img_path = join(image_dir, "src-font-images")
        self.b_path = join(image_dir, "train_images")
        self.image_filenames = [x for x in listdir(self.b_path) if is_image_file(x)]
        self.img_size = img_size
       
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        # Get Images
        image = self.transform(Image.open(join(self.b_path, self.image_filenames[index])).convert('RGB'))

        return image

    def __len__(self):
        return len(self.image_filenames)

def plot_data(loader):
    # Plot some training images
    image = next(iter(loader))
    print("image shape", image.shape)

    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("training images")      
    plt.imshow(np.transpose(vutils.make_grid(image[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()
