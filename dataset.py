from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
import random
import torch.utils.data as data
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

class ColorDataset(data.Dataset):
    #def __init__(self, loadSize, fineSize, data_root, img_root, transform=None):
    def __init__(self, args, transform=None):
        # args img_root is a txt file that has all the paths to images
        # transform is an optional transform to be applied on a sample
        self.root = args.dataset_root
        f = open(args.img_root, 'r')
        self.imgs = f.readlines()
        f.close()
        self.transform = transform

        self.loadSize = args.loadSize
        self.fineSize = args.fineSize

    def name(self):
        return 'ColorDataset'

    def __getitem__(self, index):
        img_name = os.path.join(self.root, self.imgs[index].strip())
        img = Image.open(img_name).convert('RGB')
        w,h = img.size

        # crop and resize
        # in pix2pix network
        # A is the data and B is the ground truth
        # but colorization is self-supervised
        # so I only need to return the whole image and deal with the image when do the actual training
        img = img.crop((0,0,w,h)).resize((self.loadSize, self.loadSize), Image.BICUBIC)
        w_offset = random.randint(0, max(0, self.loadSize-self.fineSize-1))
        h_offset = random.randint(0, max(0, self.loadSize-self.fineSize-1))
        img = transforms.ToTensor()(img)
        img = img[:, h_offset:h_offset+self.fineSize, w_offset:w_offset+self.fineSize]

        return {'img': img, 'img_path': img_name, 'img_idx': str(index)}


    def __len__(self):
        return len(self.imgs)
