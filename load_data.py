# Copyright 2020 by Andrey Ignatov. All Rights Reserved.

from torch.utils.data import Dataset
from torchvision import transforms
from scipy import misc
import numpy as np
import imageio
import torch
import os
import imageio
from PIL import Image
import cv2
import os

import colour

from colour_demosaicing import (
    EXAMPLES_RESOURCES_DIRECTORY,
    demosaicing_CFA_Bayer_bilinear,
    demosaicing_CFA_Bayer_Malvar2004,
    demosaicing_CFA_Bayer_Menon2007,
    mosaicing_CFA_Bayer)

to_tensor = transforms.Compose([
    transforms.ToTensor()
])

def my_resize(arr, scale):
    img = Image.fromarray(arr)
    return np.array(img.resize( ( int(img.width * scale), int(img.height*  scale) ) ) )
    
def extract_bayer_channels(raw):

    # Reshape the input bayer image

    ch_B  = raw[1::2, 1::2]
    ch_Gb = raw[0::2, 1::2]
    ch_R  = raw[0::2, 0::2]
    ch_Gr = raw[1::2, 0::2]

    RAW_combined = np.dstack((ch_B, ch_Gb, ch_R, ch_Gr))
    RAW_norm = RAW_combined.astype(np.float32) / (4 * 255)

    return RAW_norm


class LoadData(Dataset):

    def __init__(self, dataset_dir, dataset_size, dslr_scale, test=False, return_raw = False, step = 'demosaic', return_path = False):

        if test:
            self.raw_dir = os.path.join(dataset_dir, 'test', 'huawei_raw')
            self.dslr_dir = os.path.join(dataset_dir, 'test', 'canon')
            self.dataset_size = dataset_size
        else:
            self.raw_dir = os.path.join(dataset_dir, 'train', 'huawei_raw')
            self.dslr_dir = os.path.join(dataset_dir, 'train', 'canon')

        self.dataset_size = dataset_size
        self.scale = dslr_scale
        self.test = test
        
        self.return_raw = return_raw
        self.return_path = return_path
        self.step = step

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        
        raw_path = os.path.join(self.raw_dir, str(idx) + '.png')
        raw_image = np.asarray(imageio.imread(raw_path))
            
        raw_image = extract_bayer_channels(raw_image)
        raw_image = torch.from_numpy(raw_image.transpose((2, 0, 1)))

        dslr_image = imageio.imread(os.path.join(self.dslr_dir, str(idx) + ".jpg"))
        #dslr_image = misc.imread(os.path.join(self.dslr_dir, str(idx) + ".jpg"))
        dslr_image = np.asarray(dslr_image)
        #dslr_image = np.float32(misc.imresize(dslr_image, self.scale / 2.0)) / 255.0
        #print('self.scale', self.scale)
        dslr_image = np.float32(my_resize(dslr_image, self.scale / 2.0)) / 255.0
        
        dslr_image = torch.from_numpy(dslr_image.transpose((2, 0, 1)))
        
#         if self.return_path:
#                 return raw_image, dslr_image, img_isp, os.path.join(self.raw_dir, str(idx) + '.png')        
        
        if self.return_raw:
                return raw_image, dslr_image, img_isp
        
        return raw_image, dslr_image


class LoadVisualData(Dataset):

    def __init__(self, data_dir, size, scale, level, full_resolution=False):

        self.raw_dir = os.path.join(data_dir, 'test', 'huawei_full_resolution')

        self.dataset_size = size
        self.scale = scale
        self.level = level
        self.full_resolution = full_resolution
        self.test_images = os.listdir(self.raw_dir)

        if level > 1 or full_resolution:
            self.image_height = 1440
            self.image_width = 1984
        elif level > 0:
            self.image_height = 1280
            self.image_width = 1280
        else:
            self.image_height = 960
            self.image_width = 960

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):

        raw_image = np.asarray(imageio.imread(os.path.join(self.raw_dir, self.test_images[idx])))
        raw_image = extract_bayer_channels(raw_image)

        if self.level > 1 or self.full_resolution:
            raw_image = raw_image[0:self.image_height, 0:self.image_width, :]
        elif self.level > 0:
            raw_image = raw_image[80:self.image_height + 80, 352:self.image_width + 352, :]
        else:
            raw_image = raw_image[240:self.image_height + 240, 512:self.image_width + 512, :]

        raw_image = torch.from_numpy(raw_image.transpose((2, 0, 1)))

        return raw_image
