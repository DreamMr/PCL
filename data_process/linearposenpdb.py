import os
from PIL import Image
import torch.utils.data as data
import numpy as np
import torchvision.transforms as transforms
import utils
import torch
import torch.nn as nn
from data_process.DistanceCrop import DistanceCrop
from data_process.Sobel import Sobel
from data_process.transform import SequenceRandomTransform
import random
import scipy.io as scio
import glob
import cv2

class LinearPoseNpDB(data.Dataset):
    def __init__(self,config,phase):
        super(LinearPoseNpDB, self).__init__()
        self.config = config
        self.phase = phase

        #self.img_root = config['root']
        self.list_root = config[phase]
        self.sz = config['img_size']
        self.dataset = config['dataset']
        if config['pose']=='Yaw':
            self.mode = 0
        elif config['pose']=='Pitch':
            self.mode=1
        elif config['pose']=='Roll':
            self.mode=2
        elif config['pose']=='all':
            self.mode=3
        else:
            raise Exception('Not Pose mode!')

        self.eval = config['eval']

        if self.phase != 'train':
            self.transform = transforms.Compose([
                SequenceRandomTransform(),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                #transforms.Resize(self.sz),
                transforms.ToTensor()
            ])
        self.x_data, self.y_data = self.get_npz_data(self.list_root)

        
    
    def get_npz_data(self,data_path):
        files_path = glob.glob(f'{data_path}/*.npz')
        image = []
        pose = []
        for path in files_path:
            data = np.load(path)
            image.append(data['image'])
            pose.append(data['pose'])
        
        image = np.concatenate(image,0)
        pose = np.concatenate(pose,0)
        x_data = []
        y_data = []
        for i in range(pose.shape[0]):
            if np.max(pose[i,:])<=99.0 and np.min(pose[i,:])>=-99.0:
                x_data.append(image[i])
                y_data.append(pose[i])
        return np.array(x_data),np.array(y_data)
    
    
    def __getitem__(self, item):
        x = self.x_data[item]
        y = self.y_data[item]
        
        x = cv2.resize(x, (self.sz,self.sz), interpolation=cv2.INTER_CUBIC)
        x = self.transform(x)
        
        return {
                'img_normal':x,
                'label':y}

    def __len__(self):
        return self.y_data.shape[0]

class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img
