import argparse
import os 
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam

import torch
import imageio
import numpy as np
import math
import sys

from load_data import LoadData, LoadVisualData
from msssim import MSSSIM
from model import PyNET, Trans
from vgg import vgg_19, VGG16_perceptual
from utils import normalize_batch, process_command_args

import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
import foolbox.attacks as fa

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.optim import Adam
import numpy as np
import torchvision.models as models
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchattacks

from wrapper import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--command',
                        help = "peform training or inference")
    
    parser.add_argument('--data_path',
                        default = '/scratch/gpfs/yz8614/raw_images/Imagenet/',
                        help = "dataset path for evaluation")

    parser.add_argument('--mode',
                        default = 'val',
                        help = "train, validation or test")

    parser.add_argument('--batch_size',
                        default = 32,
                        help = "Inference batch size")

    parser.add_argument('--device',
                        default = '0',
                        help = "device number")

    parser.add_argument('--pt',
                        default = 2,
                        help="Maximum Perturbation Allowed")

    parser.add_argument('--no_imsave',
                        default = True,
                        help = "Store adversarial image in memory instead of locally")

    parser.add_argument('--encoder_path',
                        default = 'ckpt/encoder.pt',
                        help = "model weight for encoder (mapping from RGB to Raw)")

    parser.add_argument('--decoder_path',
                        default = 'ckpt/decoder.pt',
                        help = "model weight for decoder (mapping from RAW to RGB)")

    parser.add_argument('--fast_training',
                        default = False,
                        help = "For best performance please follow the training strategy described in the paper; If you want to perform fast training to save time, change this parameter to True")

    parser.add_argument('--level',
                        default = 1)

    parser.add_argument('--lr',
                        default = 0.001)

    parser.add_argument('--decoder_lr',
                        default = 0.00001)

    parser.add_argument('--epoch',
                        default = 100)

    parser.add_argument('--instance_norm',
                        default = True)

    parser.add_argument('--residual',
                        default = False)

    parser.add_argument('--pin_memory',
                        default = True)

    parser.add_argument('--instance_norm_level_1',
                        default = True)

    parser.add_argument('--use_percept',
                        default = False)

    parser.add_argument('--train_encoder',
                        default = False)

    parser.add_argument('--visual',
                        default = True)

    parser.add_argument('--decoder_epoch',
                        default = 10)
    parser.add_argument('--train_dataset_dir',
                        default = '/scratch/gpfs/yz8614/raw_images/',
                        help = "dataset path for training (containing the raw and rgb data)")
    
    
    args = parser.parse_args()
    
    model = Model(args)
    
    if args.command == 'eval':
        model.defense_eval()
        
    elif args.command == 'train_decoder':
        model.train_decoder_fast()
        
    elif args.command == 'train_encoder':
        model.train_encoder()
    
    else:
        print('error: command not recognized')
              
              
