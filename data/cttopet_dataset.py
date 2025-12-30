import os
from os.path import splitext
from os import listdir
from os.path import join
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
from skimage.restoration import denoise_tv_chambolle
import torchvision.transforms.functional as TF # type: ignore
from torchvision import transforms # type: ignore
import torchvision.transforms as tt # type: ignore
import torch.nn as nn
import torch.nn.functional as F
from data.base_dataset import BaseDataset
from pathlib import Path

class CTtoPETDataset(BaseDataset):
    def __init__(self, opt):
        self.mode = opt.mode
        self.preprocess_gamma = opt.preprocess_gamma
        BaseDataset.__init__(self, opt)

        if self.mode=='test':
            self.CT_dir = Path(opt.dataroot) / 'temp_folder'
            self.PET_dir = Path(opt.dataroot) / 'temp_folder'
        else:
            self.CT_dir = Path(opt.dataroot) / 'trainA'
            self.PET_dir = Path(opt.dataroot) / 'trainB'
        self.CT_dir = str(self.CT_dir.resolve())
        self.PET_dir = str(self.PET_dir.resolve())
        print(f"Debug: CT_dir = {self.CT_dir}, PET_dir = {self.PET_dir}")
        self.ids = [file for file in os.listdir(self.CT_dir)
                    if not file.startswith('.') and file.endswith('.npy')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')
    @classmethod
    def preprocessCT(cls, im, minn=-900.0, maxx=200.0, noise_std = 0):
        img_np = np.array(im)  # Shape should remain (7, 512, 512)
        # Adding Noise
        if noise_std:
            img_np += noise_std * np.random.randn(*img_np.shape)
        img_np = np.clip(img_np, minn, maxx)
        img_np = (img_np - minn) / (maxx - minn)
        print(f"CT min: {img_np.min()}, max: {img_np.max()}, shape: {img_np.shape}")
        return img_np

    # Gamma Function on PET
    @classmethod
    def preprocessPET_gamma(cls, img, gamma = 1/2, maxx = 7, noise_std=0, scale=1. ):
        img = np.array(img, dtype=np.float32)  # Ensure input is float32 for safe operations
        img /= scale  # Scale the input
        if noise_std:
            img += noise_std * np.random.randn(*img.shape)  # Add noise of the same shape
        img = np.clip(img, 0, maxx)  # Clip values to the range [0, maxx]
        img /= maxx  # Normalize to [0, 1]
        img = np.power(img, gamma)  # Apply gamma correction
        if len(img.shape) == 2:  # If the image is 2D, add a channel dimension
            img = np.expand_dims(img, axis=0)
        return img

    @classmethod
    def postprocessPET_gamma( img, gamma=1/2 ,maxx = 10.0):
        print('    gamma of {} was selected! '.format(gamma))
        img = np.array(img)
        img = np.clip(img, 0.0, 1.0)
        img = np.power(img, 1/gamma)*maxx
        return img

    # Two level Function on PET
    # @classmethod
    # def preprocessPET(cls, img, middle = 2.5 , y_axis = 0.80 , minn = 0.0, maxx = 10.0, noise_std = 0 ):
    #     img = np.array(img)
    #     img = img/100.0
    #     if noise_std:
    #         s0,s1,s2 = img.shape
    #         img = img + noise_std*np.random.randn(s0,s1,s2)
    #     imgMinMid = np.clip(img, minn, middle)
    #     imgMinMid = (imgMinMid - minn)/(middle-minn)*y_axis
    #     #
    #     imgMidMax = np.clip(img, middle, maxx)
    #     imgMidMax = (imgMidMax - middle)/(maxx-middle)*(1-y_axis) + y_axis
    #     #
    #     img = (img>=middle)*imgMidMax  + (img<middle)*imgMinMid
    #     #
    #     if len(img.shape) == 2:
    #         img = np.expand_dims(img, axis=0)
    #     return img

    @classmethod
    def edge_zero(cls, img):
        img[:,0,:] = 0
        img[:,-1,:] = 0
        img[:,:,0] = 0
        img[:,:,-1] = 0
        return img

    @classmethod
    def postprocessPET(cls, img, middle = 2.5 , y_axis = 0.85 , minn = 0.0, maxx = 10.0 ): #middle = 4   , y_axis = 0.9 , minn = 0.0, maxx = 15.0
        img = np.clip(img, minn, 1.0)
        img_L_y_axis = (img/y_axis)*middle
        m = (maxx - middle)/(1-y_axis)
        img_G_y_axis = img*m - m + maxx
        img = (img>=y_axis)*img_G_y_axis  + (img<y_axis)*img_L_y_axis
        return img

    # Data Augmentation
    def transform(self, CT, PET): # CT: (7,512,512), PET: (3,512,512)
        # Affine transformations applied to both CT and PET consistently
        if torch.rand(1) < 0.95:
            affine_params = tt.RandomAffine(0).get_params((-45, 45), (0.10, 0.10), (0.85, 1.15), (-7 , 7 ),img_size=(512,512))
        else:
            affine_params = tt.RandomAffine(0).get_params((-180, +180), (0.10, 0.10), (0.85, 1.15), (-7 , 7 ),img_size=(512,512))
        CT  = TF.affine(CT, *affine_params)
        PET = TF.affine(PET, *affine_params)
        return CT, PET

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        if self.mode == 'test':
            self.ids = np.sort(self.ids)
        idx = self.ids[i]
        PET_file = join( self.PET_dir , idx )
        CT_file = join(self.CT_dir , idx )
        # Loading
        PET = np.load(PET_file)
        CT = np.load(CT_file)
        # Normalizing
        print(f"Loaded CT shape before preprocessing: {CT.shape}")
        CT = self.preprocessCT(CT[:,:,:])
        print(f"CT shape after preprocessing: {CT.shape}")
        if not self.preprocess_gamma:
            PET = self.preprocessPET(PET[2:5,:,:]) # if 1 channel chosen then
        else:
            PET = self.preprocessPET_gamma(PET[2:5,:,:]) 
        CT = self.edge_zero(CT)
        PET = self.edge_zero(PET)
        # Data augmentation
        if self.mode == 'train':
            CT, PET = self.transform(  torch.from_numpy(CT), torch.from_numpy(PET)  )
            CT, PET = CT.type(torch.FloatTensor), PET.type(torch.FloatTensor)
        else:
            # To float before GaussianTorch(PET)
            CT = torch.from_numpy(CT).type(torch.FloatTensor)
            PET = torch.from_numpy(PET).type(torch.FloatTensor)

        return {'A': CT, 'B': PET, 'A_paths': self.CT_dir, 'B_paths': self.PET_dir, 'name':idx}
