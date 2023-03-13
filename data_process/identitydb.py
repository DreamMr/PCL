import os
from PIL import Image
import torch.utils.data as data
import numpy as np
import torchvision.transforms as transforms
import utils
import torch
import torch.nn as nn

class IdentityDB(data.Dataset):
    def __init__(self,config,phase):
        super(IdentityDB, self).__init__()

        self.config = config

        self.img_root = config['root']
        self.list_root = config[phase]
        self.sz = config['img_size']
        self.dataset = config['dataset']
        __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                            'std': [0.229, 0.224, 0.225]}

        if phase == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(self.sz),
                #transforms.RandomResizedCrop(self.sz, scale=(0.8, 1.2), ratio=(0.8, 1.2), interpolation=2),
                transforms.ToTensor()
                #transforms.Normalize(**__imagenet_stats)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(self.sz),
                #transforms.RandomResizedCrop(self.sz, scale=(0.8, 1.2), ratio=(0.8, 1.2), interpolation=2),
                transforms.ToTensor()
                # transforms.Normalize(**__imagenet_stats)
            ])

        if self.dataset == 'LFW':
            self.data_list = self.get_list_LFW(self.list_root)
            #print(len(self.data_list),self.list_root)
        elif self.dataset == 'CPLFW':
            self.data_list = self.get_list_CPLFW(self.list_root)
        else:
            raise Exception('Not Found dataset !')

    def get_list_CPLFW(self,path):
        data_list = []
        with open(path,'r') as imf:
            for i,line in enumerate(imf):
                line = line.strip()
                arr = line.split(' ')
                img_name1 = arr[0]
                img_name2 = arr[1]
                label = float(arr[2])

                data_list.append((img_name1,img_name2,label))
        return data_list

    def get_list_LFW(self,path):
        data_list = []
        with open(path,'r') as imf:
            for i, line in enumerate(imf):
                line = line.strip()
                arr = line.split('\t')
                if len(arr) == 3:
                    dir_name = arr[0]
                    img_name1 = dir_name + '_' + arr[1].zfill(4) + '.jpg'
                    img_name2 = dir_name + '_' + arr[2].zfill(4) + '.jpg'

                    img_path1 = os.path.join(dir_name,img_name1)
                    img_path2 = os.path.join(dir_name,img_name2)
                    data_list.append((img_path1,img_path2,1.))
                elif len(arr) == 4:
                    dir_name1 = arr[0]
                    img_name1 = dir_name1 + '_'+ arr[1].zfill(4) + '.jpg'
                    img_path1 = os.path.join(dir_name1,img_name1)

                    dir_name2 = arr[2]
                    img_name2 = dir_name2 + '_' + arr[3].zfill(4) + '.jpg'
                    img_path2 = os.path.join(dir_name2,img_name2)

                    data_list.append((img_path1,img_path2,-1.))
        return data_list

    def __getitem__(self, item):
        img_path1,img_path2,label = self.data_list[item]

        img1 = Image.open(os.path.join(self.img_root,img_path1)).convert('RGB')
        img2 = Image.open(os.path.join(self.img_root,img_path2)).convert('RGB')

        img_normal_1 = self.transform(img1)
        img_normal_2 = self.transform(img2)

        return {'img_normal1':img_normal_1,'img_normal2':img_normal_2,'label':label}

    def __len__(self):
        return len(self.data_list)