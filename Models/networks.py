import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.nn.functional as F
from functools import partial
import math
import torchvision.models as models
import functools
import torchvision.models as models
import random

class Generator(torch.nn.Module):
    def __init__(self,d_model):
        super(Generator, self).__init__()
        self.d_model = d_model
        
        self.fc = nn.Linear(d_model,512)
        self.d_model = 512

        up = nn.Upsample(scale_factor=2, mode='bilinear')

        dconv1 = nn.Conv2d(self.d_model, self.d_model//2, 3, 1, 1) # 2*2 512
        dconv2 = nn.Conv2d(self.d_model//2, self.d_model//2, 3, 1, 1) # 4*4 256
        dconv3 = nn.Conv2d(self.d_model//2, self.d_model//2, 3, 1, 1) # 16*16 256
        dconv4 = nn.Conv2d(self.d_model//2, self.d_model//2, 3, 1, 1) # 32 * 32 * 256
        dconv5 = nn.Conv2d(self.d_model//2, self.d_model//4, 3, 1, 1) #  64 * 64 *128
        #dconv6 = nn.Conv2d(self.d_model//4, self.d_model//8, 3, 1, 1) # 128 * 128 *32
        dconv7 = nn.Conv2d(self.d_model//4, 3, 3, 1, 1)

        # batch_norm2_1 = nn.BatchNorm2d(self.d_model//8)
        batch_norm4_1 = nn.BatchNorm2d(self.d_model//4)
        batch_norm8_4 = nn.BatchNorm2d(self.d_model//2)
        batch_norm8_5 = nn.BatchNorm2d(self.d_model//2)
        batch_norm8_6 = nn.BatchNorm2d(self.d_model//2)
        batch_norm8_7 = nn.BatchNorm2d(self.d_model//2)

        relu = nn.ReLU()
        tanh = nn.Tanh()

        self.model = torch.nn.Sequential(relu, up, dconv1, batch_norm8_4, \
                             relu, up, dconv2, batch_norm8_5, relu,
                             up, dconv3, batch_norm8_6, relu, up, dconv4,
                             batch_norm8_7, relu, up, dconv5, batch_norm4_1,
                             relu, up, dconv7, tanh)

    def forward(self,x):
        x = self.fc(x)
        x = x.unsqueeze(dim=2).unsqueeze(dim=3)
        out = self.model(x)
        return out

class selfattention(nn.Module):
    def __init__(self, inplanes):
        super(selfattention, self).__init__()

        self.interchannel = inplanes
        self.inplane = inplanes
        self.g = nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(inplanes, self.interchannel, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(inplanes, self.interchannel, kernel_size=1, stride=1, padding=0)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        b, c, h, w = x.size()
        g_y = self.g(x).view(b, c, -1)  # BXcXN
        theta_x = self.theta(x).view(b, self.interchannel, -1)
        theta_x = F.softmax(theta_x, dim=-1)  # softmax on N
        theta_x = theta_x.permute(0, 2, 1).contiguous()  # BXNXC'

        phi_x = self.phi(x).view(b, self.interchannel, -1)  # BXC'XN

        similarity = torch.bmm(phi_x, theta_x)  # BXc'Xc'

        g_y = F.softmax(g_y, dim=1)
        attention = torch.bmm(similarity, g_y)  # BXCXN
        attention = attention.view(b, c, h, w).contiguous()
        y = self.act(x + attention)
        return y

class BasicBlockNormal(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockNormal, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes,planes,3,stride,1)
        self.relu = nn.LeakyReLU(negative_slope=0.1,inplace=True)
        self.conv2 = nn.Conv2d(planes,planes,3,1,1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        #out = self.relu(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = (out + identity)
        return self.relu(out)

class FaceCycleBackboneSingle(torch.nn.Module):
    def __init__(self):
        super(FaceCycleBackboneSingle, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, 7, 2, 3, bias=True),
                                   nn.LeakyReLU(negative_slope=0.1),
                                   nn.Conv2d(64, 64, 3, 1, 1, bias=True),
                                   nn.LeakyReLU(negative_slope=0.1))

        self.layer1 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.1),
                                    selfattention(64),
                                    nn.Conv2d(64, 64, 3, 1, 1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.1))  # 64

        self.layer2_1 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1, bias=True),
                                      nn.LeakyReLU(negative_slope=0.1),
                                      selfattention(128),
                                      nn.Conv2d(128, 128, 3, 1, 1, bias=True),
                                      nn.LeakyReLU(negative_slope=0.1), )  # 64

        self.resblock1 = BasicBlockNormal(128, 128)
        self.resblock2 = BasicBlockNormal(128, 128)

        self.layer2_2 = nn.Sequential(nn.Conv2d(128, 128, 3, 2, 1, bias=True),
                                      nn.LeakyReLU(negative_slope=0.1),
                                      nn.Conv2d(128, 128, 3, 1, 1, bias=True),
                                      nn.LeakyReLU(negative_slope=0.1), )  # 64

        self.layer3_1 = nn.Sequential(nn.Conv2d(128, 256, 3, 2, 1, bias=True),
                                      nn.LeakyReLU(negative_slope=0.1),
                                      nn.Conv2d(256, 128, 3, 1, 1, bias=True),
                                      nn.LeakyReLU(negative_slope=0.1), )  # 64

        self.layer3_2 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1, bias=True),  # stride 2 for 128x128
                                      nn.LeakyReLU(negative_slope=0.1),
                                      nn.Conv2d(128, 128, 3, 1, 1, bias=True),
                                      nn.LeakyReLU(negative_slope=0.1),
                                      )  # 64
        #self.fc = nn.Identity()
        # self.exp_fc = nn.Sequential(nn.Linear(2048, 2048),
        #                             nn.ReLU(),
        #                             nn.Linear(2048, 512),
        #                             nn.BatchNorm1d(512))
        #
        # self.pose_fc = nn.Sequential(nn.Linear(2048, 2048),
        #                              nn.ReLU(),
        #                              nn.Linear(2048, 512),
        #                              nn.BatchNorm1d(512))

        #self.decoder = Generator(d_model=512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        #encoder
        '''

        :param x: [batch,3,64,64]
        :return:
        '''
        out_1 = self.conv1(x) # [batch,64,32,32]
        out_1 = self.layer1(out_1) # [batch,64,32,32]
        out_2 = self.layer2_1(out_1) # [batch,128,16,16]
        out_2 = self.resblock1(out_2) # [batch,128,16,16]
        out_2 = self.resblock2(out_2) # [batch,128,16,16]
        out_2 = self.layer2_2(out_2) # [batch,128,8,8]
        out_3 = self.layer3_1(out_2) # [batch,256,4,4]
        out_3 = self.layer3_2(out_3).view(x.size()[0],-1)
        #print(out_3.size())
        # expcode = self.fc(out_3) # [batch,256]
        #out_3 = self.fc(out_3)
        #exp_fea = self.exp_fc(out_3)
        #pose_fea = self.pose_fc(out_3)
        #fea = torch.cat([exp_fea,pose_fea],dim=1)
        #fea = exp_fea + pose_fea
        #return fea
        #return out_3
        #return exp_fea
        #return pose_fea
        #return exp_fea,pose_fea
        #return expcode
        return out_3

class FaceCycleBackbone(torch.nn.Module):
    def __init__(self):
        super(FaceCycleBackbone, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, 7, 2, 3, bias=True),
                                   nn.LeakyReLU(negative_slope=0.1),
                                   nn.Conv2d(64, 64, 3, 1, 1, bias=True),
                                   nn.LeakyReLU(negative_slope=0.1))

        self.layer1 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.1),
                                    selfattention(64),
                                    nn.Conv2d(64, 64, 3, 1, 1, bias=True),
                                    nn.LeakyReLU(negative_slope=0.1))  # 64

        self.layer2_1 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1, bias=True),
                                      nn.LeakyReLU(negative_slope=0.1),
                                      selfattention(128),
                                      nn.Conv2d(128, 128, 3, 1, 1, bias=True),
                                      nn.LeakyReLU(negative_slope=0.1), )  # 64

        self.resblock1 = BasicBlockNormal(128, 128)
        self.resblock2 = BasicBlockNormal(128, 128)

        self.layer2_2 = nn.Sequential(nn.Conv2d(128, 128, 3, 2, 1, bias=True),
                                      nn.LeakyReLU(negative_slope=0.1),
                                      nn.Conv2d(128, 128, 3, 1, 1, bias=True),
                                      nn.LeakyReLU(negative_slope=0.1), )  # 64

        self.layer3_1 = nn.Sequential(nn.Conv2d(128, 256, 3, 2, 1, bias=True),
                                      nn.LeakyReLU(negative_slope=0.1),
                                      nn.Conv2d(256, 128, 3, 1, 1, bias=True),
                                      nn.LeakyReLU(negative_slope=0.1), )  # 64

        # # self.layer3_2 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1, bias=True),  # stride 2 for 128x128
        # #                                   nn.LeakyReLU(negative_slope=0.1),
        # #                                   nn.Conv2d(256, 128, 3, 1, 1, bias=True),
        # #                                   nn.LeakyReLU(negative_slope=0.1),
        # #                                   )  # 64
        self.layer3_2_exp = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1, bias=True),  # stride 2 for 128x128
                                      nn.LeakyReLU(negative_slope=0.1),
                                      nn.Conv2d(128, 128, 3, 1, 1, bias=True),
                                      nn.LeakyReLU(negative_slope=0.1),
                                      )  # 64

        self.layer3_2_pose = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1, bias=True),  # stride 2 for 128x128
                                          nn.LeakyReLU(negative_slope=0.1),
                                          nn.Conv2d(128, 128, 3, 1, 1, bias=True),
                                          nn.LeakyReLU(negative_slope=0.1))  # 64


        #self.fc = nn.Identity()
        # self.exp_fc = nn.Sequential(nn.Linear(2048, 2048),
        #                             nn.ReLU(),
        #                             nn.Linear(2048, 512),
        #                             nn.BatchNorm1d(512))
        #
        # self.pose_fc = nn.Sequential(nn.Linear(2048, 2048),
        #                              nn.ReLU(),
        #                              nn.Linear(2048, 512),
        #                              nn.BatchNorm1d(512))

        #self.decoder = Generator(d_model=512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        #encoder
        '''

        :param x: [batch,3,64,64]
        :return:
        '''
        out_1 = self.conv1(x) # [batch,64,32,32]
        out_1 = self.layer1(out_1) # [batch,64,32,32]
        out_2 = self.layer2_1(out_1) # [batch,128,16,16]
        out_2 = self.resblock1(out_2) # [batch,128,16,16]
        out_2 = self.resblock2(out_2) # [batch,128,16,16]
        out_2 = self.layer2_2(out_2) # [batch,128,8,8]
        out_3 = self.layer3_1(out_2) # [batch,256,4,4]

        out_3_exp = self.layer3_2_exp(out_3) # [batch,128,4,4]
        out_3_exp = out_3_exp.view(x.size()[0],-1) # [batch,2048]

        out_3_pose = self.layer3_2_pose(out_3)
        out_3_pose = out_3_pose.view(x.size()[0],-1)

        out_3 = out_3.view(x.size()[0],-1)
        #out_3 = self.layer3_2(out_3).view(x.size()[0],-1)
        #print(out_3.size())
        # expcode = self.fc(out_3) # [batch,256]
        #out_3 = self.fc(out_3)
        #exp_fea = self.exp_fc(out_3)
        #pose_fea = self.pose_fc(out_3)
        #fea = torch.cat([exp_fea,pose_fea],dim=1)
        #fea = exp_fea + pose_fea
        #return fea
        #return out_3
        #return exp_fea
        #return pose_fea
        #return exp_fea,pose_fea
        #return expcode
        return out_3, out_3_exp,out_3_pose
        #return out_3_exp

class Projection(nn.Module):
    def __init__(self,in_dim=2048,out_dim=512):
        super(Projection,self).__init__()
        self.linear1 = nn.Linear(in_dim,in_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_dim,out_dim)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self,x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.bn(out)
        return out

class ExpPoseModel(nn.Module):
    def __init__(self):
        super(ExpPoseModel, self).__init__()

        self.encoder = FaceCycleBackbone()
        #self.encoder = resnet18(num_classes=2048)
        #self.encoder = resnet34()
        #self.encoder = resnet50(num_classes=2048)

        self.exp_fc = nn.Sequential(nn.Linear(2048, 2048),
                                         nn.ReLU(),
                                         nn.Linear(2048,512),
                                         nn.BatchNorm1d(512))

        self.pose_fc = nn.Sequential(nn.Linear(2048, 2048),
                                         nn.ReLU(),
                                         nn.Linear(2048,512),
                                         nn.BatchNorm1d(512))

        self.decoder = Generator(d_model=2048)

    def forward(self,exp_img=None,normal_img=None,flip_img=None,exp_image_pose_neg=None,recon_only=False,state='pfe'):
        if not recon_only:
            if state == 'pfe':
                _,normal_exp_fea,normal_pose_fea = self.encoder(normal_img)
                _,flip_exp_fea,flip_pose_fea = self.encoder(flip_img)

                recon_normal_exp_flip_pose_fea = F.normalize(normal_exp_fea+flip_pose_fea, dim=1)
                recon_flip_exp_normal_pose_fea = F.normalize(flip_exp_fea+normal_pose_fea, dim=1)
                recon_normal_exp_normal_pose_fea = F.normalize(normal_exp_fea+normal_pose_fea, dim=1)
                recon_flip_exp_flip_pose_fea = F.normalize(flip_exp_fea+flip_pose_fea,dim=1)

                recon_normal_exp_flip_pose_img = self.decoder(recon_normal_exp_flip_pose_fea)
                recon_flip_exp_normal_pose_img = self.decoder(recon_flip_exp_normal_pose_fea)
                recon_normal_exp_normal_pose_img = self.decoder(recon_normal_exp_normal_pose_fea)
                recon_flip_exp_flip_psoe_img = self.decoder(recon_flip_exp_flip_pose_fea)

                return normal_exp_fea,normal_pose_fea,flip_exp_fea,flip_pose_fea, recon_normal_exp_flip_pose_img,recon_flip_exp_normal_pose_img,recon_normal_exp_normal_pose_img,recon_flip_exp_flip_psoe_img
            elif state == 'exp':
                _,exp_fea,_ = self.encoder(exp_img)
                exp_fea_fc = self.exp_fc(exp_fea)
                return exp_fea_fc
            elif state == 'pose':
                _,_,pose_fea = self.encoder(exp_img)
                pose_fea_fc = self.pose_fc(pose_fea)

                # todo check
                _,_,pose_neg_fea = self.encoder(exp_image_pose_neg)
                pose_neg_fea_fc = self.pose_fc(pose_neg_fea)
                return pose_fea_fc,pose_neg_fea_fc
        else:
            _,normal_exp_fea, normal_pose_fea = self.encoder(normal_img)
            _,flip_exp_fea, flip_pose_fea = self.encoder(flip_img)

            recon_normal_exp_flip_pose_fea = F.normalize(normal_exp_fea + flip_pose_fea, dim=1)
            recon_flip_exp_normal_pose_fea = F.normalize(flip_exp_fea + normal_pose_fea, dim=1)
            recon_normal_exp_normal_pose_fea = F.normalize(normal_exp_fea + normal_pose_fea, dim=1)
            recon_flip_exp_flip_pose_fea = F.normalize(flip_exp_fea + flip_pose_fea, dim=1)

            recon_normal_exp_flip_pose_img = self.decoder(recon_normal_exp_flip_pose_fea)
            recon_flip_exp_normal_pose_img = self.decoder(recon_flip_exp_normal_pose_fea)
            recon_normal_exp_normal_pose_img = self.decoder(recon_normal_exp_normal_pose_fea)
            recon_flip_exp_flip_psoe_img = self.decoder(recon_flip_exp_flip_pose_fea)

            return normal_exp_fea,normal_pose_fea,flip_exp_fea,flip_pose_fea, recon_normal_exp_flip_pose_img,recon_flip_exp_normal_pose_img,recon_normal_exp_normal_pose_img,recon_flip_exp_flip_psoe_img

