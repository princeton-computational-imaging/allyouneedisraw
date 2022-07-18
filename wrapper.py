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

class Model:

    def __init__(self, args):
        self.args = args

        if args.mode != 'train':
            self.load_model()
            model, encoder, decoder = self.model, self.encoder, self.decoder 
            preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
            self.fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing = preprocessing)

            adv_images_infer = []
            reconstruct_acc = []
            label = []

            for batch_i, (images, target) in enumerate(self.init_utils()):

                images = images.cuda()
                target = target.cuda()
                images = self.invTrans(images)

                output = self.fmodel(images)
                true_index = (output.argmax(1) == target) 

                adv_images, _, success = self.attack(self.fmodel, images, target, epsilons = [args.pt / 255])
                adv_images = adv_images[0]

                adv_images_infer.append(adv_images[true_index].cpu())
                label.append(target[true_index].cpu())


            assert args.no_imsave == True
            self.adv_images_infer = torch.cat(adv_images_infer, 0)
            self.label = torch.cat(label, 0)

    def train_decoder_fast(self):
        
        # train Raw to RBG Mapping
        self.inin_utils_training()
        MSE_loss = self.MSE_loss
        args = self.args
        train_loader, level = self.train_loader, args.level
        MS_SSIM = MSSSIM()
        device = 'cuda'
        VGG_19 = vgg_19(device)
        
        
        assert args.fast_training
        
        generator = PyNET(level = args.level, 
                          instance_norm = args.instance_norm, 
                          instance_norm_level_1 = args.instance_norm_level_1,
                          residual = args.residual).to(device)
        generator = torch.nn.DataParallel(generator)

        optimizer = Adam(params = generator.parameters(), lr = args.decoder_lr)   
        
        for epoch in range(args.decoder_epoch):
            
            print('------------------ start training the decoder ------------------------')
            torch.cuda.empty_cache()
            train_iter = iter(train_loader)

            for i in range(len(train_loader)):

                optimizer.zero_grad()
                x, y = next(train_iter)

                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                enhanced = generator(x)

                loss_mse = MSE_loss(enhanced, y)

                if level < 5:
                    enhanced_vgg = VGG_19(normalize_batch(enhanced))
                    target_vgg = VGG_19(normalize_batch(y))
                    loss_content = MSE_loss(enhanced_vgg, target_vgg)

                if level == 5 or level == 4:
                    total_loss = loss_mse
                if level == 3 or level == 2:
                    total_loss = loss_mse * 10 + loss_content
                if level == 1:
                    total_loss = loss_mse * 10 + loss_content
                if level == 0:
                    loss_ssim = MS_SSIM(enhanced, y)
                    total_loss = loss_mse + loss_content + (1 - loss_ssim) * 0.4

                total_loss.backward()
                optimizer.step()

                if i == 0:
                    
                    if args.visual:
                        plt.imshow(enhanced[0].permute(1,2,0).cpu().detach().numpy())
                        plt.show()
                        
                    generator.eval().cpu()
                    torch.save(generator.state_dict(), "ckpt/decoder_level_" + str(level)  + '_' + str(args.residual) +"_epoch_" + str(epoch) + ".pth")
                    generator.to(device).train()     
                    
        return 
        
    def train_encoder(self):
        
        # train RGB to Raw Mapping
        
        self.inin_utils_training()
        MSE_loss = self.MSE_loss
        args = self.args
        train_loader = self.train_loader
        batch_size = args.batch_size
        
        encoder = Trans()
        encoder.cuda()
        encoder.train()

        optimizer = Adam(params = encoder.parameters(), lr = args.lr)

        interation = 0
        loss = []

        for epoch in range(0, args.epoch):
            
            print('-------------- starting training the encoder ----------------')
            
            torch.cuda.empty_cache()
            train_iter = iter(train_loader)   

            for i in range(len(train_loader)):
                optimizer.zero_grad()

                intermediate, rgb = next(train_iter)
                intermediate = intermediate.cuda().float()
                rgb = rgb.cuda().float()        

                noise = (np.random.normal(0, 1, batch_size).reshape((batch_size,1,1,1)) * np.random.normal(0.0, 0.1, (batch_size, 3, 224, 224)))
                rgb_noise = rgb + torch.from_numpy(noise).cuda().float()       
                pred_intermediate = encoder(rgb)  
                pred_intermediate_noise = encoder(rgb_noise)

                total_loss = MSE_loss(pred_intermediate, intermediate) +  MSE_loss(pred_intermediate_noise, intermediate)

                total_loss.backward()
                optimizer.step()

                if interation % 100 == 0:
                    print('interation', interation, 'loss', total_loss.item())

                interation += 1

            print('epoch', epoch, 'loss', total_loss.item())

            if epoch % 3 == 0:
                torch.save(encoder.state_dict(), 'ckpt/encoder_' + str(epoch) + '.pt')
                
        return
        

    def percept_loss(self, syn_img, img, MSE_loss):

        MSE_loss = self.MSE_loss
            
        syn0, syn1, syn2, syn3 = self.perceptual(syn_img)
        r0, r1, r2, r3 = self.perceptual(img)
        mse = MSE_loss(syn_img,img)

        per_loss = 0
        per_loss += MSE_loss(syn0,r0)
        per_loss += MSE_loss(syn1,r1)
        per_loss += MSE_loss(syn2,r2)
        per_loss += MSE_loss(syn3,r3)

        return mse, per_loss

    
    def inin_utils_training(self):
        
        self.init_utils()
        args = self.args
        self.dslr_scale = float(1) / (2 ** (args.level - 1))
        
        train_dataset = LoadData(args.train_dataset_dir, 46839, self.dslr_scale, test=False)
        
        batch_size = args.batch_size if args.train_encoder else 2
        
        self.train_loader = DataLoader(dataset=train_dataset, batch_size = batch_size, shuffle=True, num_workers=1,
                                  pin_memory=True, drop_last=True)
        
        self.to_image = transforms.Compose([transforms.ToPILImage()])
        
        return 


    def init_utils(self):
        
        args = self.args
        
        if args.use_percept:
            self.perceptual = VGG16_perceptual().cuda()
            self.perceptual.eval()
            
        self.MSE_loss = nn.MSELoss(reduction="mean")
        self.to_image = transforms.Compose([transforms.ToPILImage()])
        self.attack = fa.FGSM()
        
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                 std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                            transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                 std = [ 1., 1., 1. ]),
                           ])
            
        dataset = datasets.ImageFolder(
            args.data_path + args.mode,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ]))

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size = args.batch_size, shuffle = True,
            pin_memory = args.pin_memory)
        
        return loader

    def load_model(self):
        args = self.args
        
        encoder = Trans()
        encoder.cuda()
        encoder.eval()
        encoder.load_state_dict(torch.load(args.encoder_path))

        decoder = PyNET(level = args.level, instance_norm = args.instance_norm, instance_norm_level_1 = args.instance_norm_level_1).cuda() #to(device)
        decoder = torch.nn.DataParallel(decoder)
        decoder.cuda()
        decoder.eval()
        decoder.load_state_dict(torch.load(args.decoder_path), strict=True)

        model =  models.resnet101(pretrained = True)
        model.cuda()
        model.eval()
        
        self.encoder = encoder
        self.decoder = decoder
        self.model = model
        
        return 
        
    def defense_eval(self):
        
        args = self.args
        infer_batch_size = args.batch_size
        adv_images_infer = self.adv_images_infer
        label = self.label
        model, encoder, decoder, fmodel = self.model, self.encoder, self.decoder, self.fmodel
        reconstruct_acc = []
        
        for i in range(len(adv_images_infer) // infer_batch_size):
            print(i)
            start, end = infer_batch_size * i, min(infer_batch_size * i + infer_batch_size, len(adv_images_infer))
            adv_images = adv_images_infer[start:end].cuda()
            ecd_adv = encoder(adv_images)

            recons_img = []

            for i in range(len(ecd_adv)):
                recons_img.append(decoder(ecd_adv[i:i+1].detach()) .cpu().detach().numpy())
                torch.cuda.empty_cache()

            recons_img = np.concatenate(recons_img)
            recons_img = torch.from_numpy(recons_img).cuda()

            output = fmodel(recons_img) 
            reconstruct_acc.append((output.argmax(1) == label[start:end].cuda()).cpu().detach().numpy().mean())
            self.recons_img = recons_img

        print('Defense Accuracy:', np.mean(reconstruct_acc))
