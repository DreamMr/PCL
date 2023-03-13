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
import random

class PfeDB(data.Dataset):
    def __init__(self,config,phase):
        super(PfeDB,self).__init__()

        self.config = config
        self.phase = phase

        self.img_root = config['root']
        self.list_root = config[phase]
        self.sz = config['img_size']
        self.dataset = config['dataset']

        self.flip = transforms.RandomHorizontalFlip(p=1.0)
        # self.flip = transforms.Compose([transforms.RandomHorizontalFlip(p=1.),
        #     transforms.RandomRotation(40)
        # ])
        self.normal_data_transform = transforms.Compose([transforms.Resize((self.sz,self.sz)),
                                                         transforms.ToTensor(),
                                                         ])

        if self.dataset == 'BU3D':
            self.data_list = self.get_list_BU3D(self.list_root)
        elif self.dataset == 'RAFDB':
            self.data_list = self.get_list_rafdb(self.list_root)
        elif self.dataset == 'VOX':
            self.data_list = self.get_list_vox(self.list_root)
        elif self.dataset == 'CASIA':
            self.data_list = self.get_list_casia(self.list_root)
        else:
            raise Exception('Not Found dataset !')
            
    def get_list_casia(self,path):
        data_list = []
        tmp = os.listdir(path)
        for f in tmp:
            data_list.append((os.path.join(self.img_root,f),0))
        return data_list

    def get_list_rafdb(self,path):
        data_list = []
        with open(path, 'r') as imf:
            for i, line in enumerate(imf):
                arr = line.strip().split(' ')
                img_path = arr[0].split('.')[0] + '_aligned.jpg'
                label = arr[1]
                data_list.append((img_path, int(label)-1))
        return data_list

    def get_list_BU3D(self,path):
        data_list = []
        with open(path,'r') as imf:
            for i, line in enumerate(imf):
                arr = line.strip().split(',')
                img_path = arr[0]
                #exp_label = arr[1]
                pose_label = arr[2]
                data_list.append((img_path,int(pose_label)))

        return data_list

    def get_list_vox(self,path):
        data_list = []
        with open(path,'r') as imf:
            for i, line in enumerate(imf):
                line = line.strip()
                data_list.append((os.path.join(self.img_root,line),0))

        return data_list

    def get_vox_image(self,dir_path):

        file_list = os.listdir(dir_path)
        file_name = random.choice(file_list)
        file_path = os.path.join(dir_path,file_name)

        img_list = os.listdir(file_path)
        img_name = random.choice(img_list)
        img_path = os.path.join(file_path,img_name)
        img = Image.open(img_path).convert('RGB')
        return img
        
    def get_casia_image(self,dir_path):
        file_list = os.listdir(dir_path)
        img_name = random.choice(file_list)
        
        img_path = os.path.join(dir_path,img_name)
        img = Image.open(img_path).convert('RGB')
        return img

    def __getitem__(self, item):
        img_path, label = self.data_list[item]
        if self.dataset == 'VOX':
            img = self.get_vox_image(img_path)
        elif self.dataset == 'CASIA':
            img = self.get_casia_image(img_path)
        else:
            img = Image.open(os.path.join(self.img_root, img_path)).convert('RGB')

        img_flip = self.flip(img)

        img_normal = self.normal_data_transform(img)
        img_flip_normal = self.normal_data_transform(img_flip)

        return {
            'img_normal':img_normal,
            'img_flip':img_flip_normal
        }

    def __len__(self):
        return len(self.data_list)