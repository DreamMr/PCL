import torch
import torch.nn as nn
import torchvision
import numpy as np
import torch.nn.functional as F
from functools import partial
import math
from facenet_pytorch import InceptionResnetV1
import torchvision.models as models
import functools
import torchvision.models as models
import random
from Models.BaseModel import BaseModel
#from Models.resnet import *

class Generator(torch.nn.Module):
    def __init__(self,d_model):
        super(Generator, self).__init__()
        self.d_model = d_model

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
                                      nn.Conv2d(256, 256, 3, 1, 1, bias=True),
                                      nn.LeakyReLU(negative_slope=0.1), )  # 64

        self.layer3_2 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1, bias=True),  # stride 2 for 128x128
                                      nn.LeakyReLU(negative_slope=0.1),
                                      nn.Conv2d(256, 128, 3, 1, 1, bias=True),
                                      nn.LeakyReLU(negative_slope=0.1))  # 64


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
        out_3 = self.layer3_2(out_3) # [batch,128,4,4]
        out_3 = out_3.view(x.size()[0],-1) # [batch,2048]
        #print(out_3.size())
        # expcode = self.fc(out_3) # [batch,256]
        #out_3 = self.fc(out_3)
        #exp_fea = self.exp_fc(out_3)
        #pose_fea = self.pose_fc(out_3)
        #fea = torch.cat([exp_fea,pose_fea],dim=1)
        #fea = exp_fea + pose_fea
        #return fea
        return out_3
        #return exp_fea
        #return pose_fea
        #return exp_fea,pose_fea
        #return expcode

class ExpPoseModel(nn.Module):
    def __init__(self):
        super(ExpPoseModel, self).__init__()

        self.encoder = FaceCycleBackbone()
        #self.encoder = resnet18()
        #self.encoder = resnet34()
        #self.encoder = resnet50()

        self.exp_fc = nn.Sequential(nn.Linear(2048, 2048),
                                         nn.ReLU(),
                                         nn.Linear(2048,512),
                                         nn.BatchNorm1d(512))

        self.pose_fc = nn.Sequential(nn.Linear(2048, 2048),
                                         nn.ReLU(),
                                         nn.Linear(2048,512),
                                         nn.BatchNorm1d(512))

        self.decoder = Generator(d_model=512)
        
        

    def forward(self,exp_img,normal_img,flip_img):
        #exp_fea = self.exp_encoder_fc(self.exp_encoder(exp_img))
        #pose_fea = self.pose_encoder_fc(self.pose_encoder(pose_img))
        fea = self.encoder(exp_img)
        exp_fea = self.exp_fc(fea)
        pose_fea = self.pose_fc(fea)

        normal_fea = self.encoder(normal_img)
        flip_fea = self.encoder(flip_img)

        normal_exp_fea_fc = self.exp_fc(normal_fea)
        flip_exp_fea_fc = self.exp_fc(flip_fea)

        normal_pose_fea_fc = self.pose_fc(normal_fea)
        flip_pose_fea_fc = self.pose_fc(flip_fea)

        ########### test
        recon_normal_exp_flip_pose_fea = F.normalize(normal_exp_fea_fc + flip_pose_fea_fc, dim=1)
        recon_flip_exp_normal_pose_fea = F.normalize(flip_exp_fea_fc+normal_pose_fea_fc, dim=1)
        recon_normal_exp_normal_exp_fea = F.normalize(normal_exp_fea_fc + normal_exp_fea_fc, dim=1)
        recon_normal_pose_normal_pose_fea = F.normalize(normal_pose_fea_fc + normal_pose_fea_fc,dim=1)
        recon_flip_pose_flip_pose_fea = F.normalize(flip_pose_fea_fc + flip_pose_fea_fc,dim=1)

        recon_normal_exp_flip_pose_img = self.decoder(recon_normal_exp_flip_pose_fea)
        recon_flip_exp_normal_pose_img = self.decoder(recon_flip_exp_normal_pose_fea)
        recon_normal_exp_normal_exp_img = self.decoder(recon_normal_exp_normal_exp_fea)
        recon_normal_pose_normal_pose_img = self.decoder(recon_normal_pose_normal_pose_fea)
        recon_flip_pose_flip_pose_img = self.decoder(recon_flip_pose_flip_pose_fea)
        #
        return recon_normal_exp_flip_pose_img,recon_flip_exp_normal_pose_img,recon_normal_exp_normal_exp_img,recon_normal_pose_normal_pose_img,recon_flip_pose_flip_pose_img

class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):
        # Zero mean
        input1_mean = torch.mean(input1, dim=0, keepdims=True)
        input2_mean = torch.mean(input2, dim=0, keepdims=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        #diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))
        diff_loss = torch.mean((input1_l2 * input2_l2).sum(dim=1).pow(2))

        return diff_loss

class ExpPose(BaseModel):
    def __init__(self,config):
        BaseModel.__init__(self,config)

        self.temperature = config['T']
        self.batch_size = config['batch_size']
        self.neg_alpha = config['neg_alpha']
        self.pose_alpha = config['pose_alpha']

        if not config['eval'] and not config['t_sne']:
            self.model = ExpPoseModel()
        else:
            #face_cycle_backbone = FaceCycleBackbone()
            #face_cycle_backbone = SimCLRNetwork()
            face_cycle_backbone = FaceCycleBackbone()
            self.model = face_cycle_backbone.cuda()

        if config['continue_train']:
            self.model = torch.nn.DataParallel(self.model).cuda()
            self.model.load_state_dict(torch.load(config['load_model'])['state_dict'])
            print('load continue model !')
        elif config['eval'] or (config['t_sne'] != None and config['t_sne']):
            #self.model.last_linear = torch.nn.Identity()
            # self.model.fc = torch.nn.Identity()
            # # self.model.fc = nn.Sequential(nn.Linear(2048, 2048),
            # #                              nn.ReLU(),
            # #                              nn.Linear(2048,512),
            # #                              nn.BatchNorm1d(512))
            if config['eval_mode'] == 'exp': # exp
                state_dict = torch.load(config['load_model'])['state_dict']
                for k in list(state_dict.keys()):
                    if k.startswith('module.encoder'):
                        state_dict[k[len("module.encoder."):]] = state_dict[k]
                    elif k.startswith('module.exp_fc') or k.startswith('module.pose_fc'):
                        state_dict[k[len("module."):]] = state_dict[k]
                #     # if k.startswith('module.exp_fc'):
                #     #     state_dict[k[len("module.exp_"):]] = state_dict[k]
                #     del state_dict[k]
                    # if k.startswith('module.exp_encoder'):
                    #     state_dict[k[len("module.exp_encoder."):]] = state_dict[k]
                    del state_dict[k]
            ######## simclr
            # state_dict = torch.load(config['load_model'])['state_dict']
            # if config['eval_mode'] == 'exp':
            #     self.model = torch.nn.DataParallel(self.model).cuda()
            elif config['eval_mode'] == 'pose': # pose
                state_dict = torch.load(config['load_model'])['state_dict']
                for k in list(state_dict.keys()):
                    if k.startswith('module.encoder'):
                        state_dict[k[len("module.encoder."):]] = state_dict[k]
                    del state_dict[k]
                print('pose loaded!')
            elif config['eval_mode'] == 'face_cycle':
                state_dict = torch.load(config['load_model'])['codegeneration']
                for k in list(state_dict.keys()):
                    if k.startswith('expresscode.'):
                        del state_dict[k]
            elif config['eval_mode'] == 'TCAE':
                self.model.fc = nn.Sequential(nn.Linear(32768, 2048),
                                            nn.ReLU(),
                                            nn.Linear(2048,2048),
                                            nn.BatchNorm1d(2048))
                state_dict = torch.load(config['load_model'])['state_dict']
                for k in list(state_dict.keys()):
                    if k.startswith('encoder.exp.'):
                        state_dict['fc.' + k[len("encoder.exp.")]] = state_dict[k]
                    if k.startswith('encoder.'):
                        state_dict[k[len("encoder."):]] = state_dict[k]
                    del state_dict[k]

            #self.model = self.model.cuda()
            #self.model = torch.nn.DataParallel(self.model).cuda()
            msg = self.model.load_state_dict(state_dict,strict=False)
            assert set(msg.missing_keys) == set()
            print('load model !')
        else:
            self.model = torch.nn.DataParallel(self.model).cuda()

        self.criterion = nn.CrossEntropyLoss().cuda()
        self.recon_criterion = nn.L1Loss().cuda()
        self.diff_loss = DiffLoss().cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=config['lr'],weight_decay=config['wd'])
        self.scaler = GradScaler()

    def get_lambda(self,epoch):
        def sigmoid(x):
            if x >= 0:
                z = math.exp(-x)
                sig = 1 / (1 + z)
                return sig
            else:
                z = math.exp(x)
                sig = z / (1 + z)
                return sig

        def exp_decay(x):
            z = math.exp(-x)
            z = max(1e-10,z)
            return z

        if epoch < 200:
            lam = (sigmoid(epoch/5.0) - 0.5) * 2.0
        else:
            lam = exp_decay((epoch-200)/5.0)

        return lam


    def optimize_parameters(self,data):
        self.model.train()

        img_normal = data['img_normal'].cuda()
        img_flip = data['img_flip'].cuda()
        cur_epoch = data['epoch']

        with autocast():
            exp_fea, pose_fea, recon_normal_exp_flip_pose_img, recon_flip_exp_normal_pose_img,recon_normal_exp_normal_pose_img = self.forward(data)

            exp_logits,exp_labels = self.neg_inter_info_nce_loss(exp_fea)
            exp_contra_loss = self.criterion(exp_logits,exp_labels)

            pose_logits,pose_labels = self.neg_inter_info_nce_loss(pose_fea)
            pose_contra_loss = self.criterion(pose_logits,pose_labels)

            recon_normal_loss = self.recon_criterion(recon_flip_exp_normal_pose_img,img_normal) # || s-s'||
            #recon_normal_loss = 0.
            recon_flip_loss = self.recon_criterion(recon_normal_exp_flip_pose_img,img_flip) # ||f-f'||
            #recon_flip_loss = 0.
            recon_orin_loss = self.recon_criterion(recon_normal_exp_normal_pose_img,img_normal) # ||s-s''||
            #recon_orin_loss = 0.

            recon_weight = self.get_lambda(cur_epoch)


            diff_loss = self.diff_loss(exp_fea,pose_fea)
            #diff_loss = 0.

            loss = exp_contra_loss + pose_contra_loss * self.pose_alpha + diff_loss + recon_weight * (recon_normal_loss + recon_flip_loss + recon_orin_loss)

        exp_acc1,exp_acc5 = utils.accuracy(exp_logits,exp_labels,(1,5))
        pose_acc1,pose_acc5 = utils.accuracy(pose_logits,pose_labels,(1,5))

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        print_img = torch.cat([img_normal[:2],img_flip[:2],recon_flip_exp_normal_pose_img[:2],recon_normal_exp_flip_pose_img[:2],recon_normal_exp_normal_pose_img[:2]],dim=3)

        return {'train_acc1_exp':exp_acc1,'train_acc5_exp':exp_acc5,'train_loss':loss,
                'train_acc1_pose':pose_acc1,'train_acc5_pose':pose_acc5,
                'train_diff_loss': diff_loss,
                'train_exp_contra_loss':exp_contra_loss,
                'train_pose_contra_loss':pose_contra_loss,
                'train_recon_normal_loss':recon_normal_loss,
                'train_recon_flip_loss':recon_flip_loss,
                'train_recon_orin_loss':recon_orin_loss,
                'train_print_img':print_img,'recon_weight':recon_weight}

    def neg_inter_info_nce_loss(self, features):

        # time_start = time.time()

        b, dim = features.size()

        labels = torch.cat([torch.arange(b // 2) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        labels_flag = (1 - labels).bool()
        features_expand = features.expand((b, b, dim))  # 512 * 512 * dim
        fea_neg_li = list(features_expand[labels_flag].chunk(b, dim=0))
        fea_neg_tensor = torch.stack(fea_neg_li, dim=0)  # 512 * 510 * dim

        # time_alpha = time.time()
        neg_mask = np.random.beta(self.neg_alpha, self.neg_alpha,
                                  size=(fea_neg_tensor.shape[0], fea_neg_tensor.shape[1]))
        time_alpha_finish = time.time()
        # print('cost alpha time: {}'.format(time_alpha_finish - time_alpha))
        if isinstance(neg_mask, np.ndarray):
            neg_mask = torch.from_numpy(neg_mask).float().cuda()
            neg_mask = neg_mask.unsqueeze(dim=2)
        indices = torch.randperm(fea_neg_tensor.shape[1])
        fea_neg_tensor = fea_neg_tensor * neg_mask + (1 - neg_mask) * fea_neg_tensor[:, indices]

        features = F.normalize(features, dim=1)
        q, k = features.chunk(2, dim=0)
        fea_neg_tensor = F.normalize(fea_neg_tensor, dim=2)

        pos = torch.cat(
            [torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1), torch.einsum('nc,nc->n', [k, q]).unsqueeze(-1)], dim=0)

        fea_neg_tensor = fea_neg_tensor.transpose(2, 1)
        neg = torch.bmm(features.view(b, 1, -1), fea_neg_tensor).view(b, -1)

        logits = torch.cat([pos, neg], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        logits = logits / self.temperature
        # print('cost time: {}'.format(time.time() - time_start))

        return logits, labels

    def forward(self,data):
        exp_images = data['exp_images']
        img_normal = data['img_normal'].cuda()
        img_flip = data['img_flip'].cuda()
        exp_images = torch.cat(exp_images,dim=0).cuda()
        exp_fea, pose_fea, recon_normal_exp_flip_pose_img, recon_flip_exp_normal_pose_img,recon_normal_exp_normal_pose_img = self.model(exp_images,img_normal,img_flip)
        return exp_fea, pose_fea, recon_normal_exp_flip_pose_img, recon_flip_exp_normal_pose_img,recon_normal_exp_normal_pose_img

    def linear_forward(self,data):
        img = data['img_normal'].cuda()
        fea = self.model(img)
        return fea

    def linear_forward_id(self,data):
        img1 = data['img_normal1'].cuda()
        fea1 = self.model(img1)

        img2 = data['img_normal2'].cuda()
        fea2 = self.model(img2)

        return fea1,fea2

    def linear_eval_id(self,data):
        self.model.eval()
        with torch.no_grad():
            fea1,fea2 = self.linear_forward_id(data)
        return fea1,fea2

    def metric_better(self,cur,best):
        ans = best
        flag = False
        if best == None or cur < best:
            flag = True
            ans = cur
        return flag,ans

    def eval(self,data):
        pass

    def linear_eval(self,data):
        self.model.eval()
        with torch.no_grad():
            fea = self.linear_forward(data)
        return fea

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    model = ExpPoseModel()
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load('./checkpoints/visualize_rafdb/best.pth')['state_dict'])
    rafdb_root = r'./dataset/RAFDB/img/aligned/'
    save_root = r'./save_imgs/'
    model = model.eval()

    from torchvision.transforms import transforms

    trans = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor()
    ])

    flip = transforms.RandomHorizontalFlip(1.0)
    to_img = transforms.Compose([transforms.ToPILImage(),transforms.Resize((100,100))])
    #gray = transforms.Grayscale(num_output_channels=3)

    from PIL import Image
    path_list = os.listdir(rafdb_root)
    img_list = []
    for p in path_list:
        if p.startswith('test'):
            img_list.append(p)
    
    count = 0
    for img_name in img_list:
        img = Image.open(os.path.join(rafdb_root,img_name)).convert('RGB')

        img_flip = flip(img)
        tensor_flip = trans(img_flip).unsqueeze(dim=0).cuda()
        tensor_normal = trans(img).unsqueeze(dim=0).cuda()

        recon_normal_exp_flip_pose_img,recon_flip_exp_normal_pose_img,recon_normal_exp_normal_pose_img,recon_normal_pose_normal_pose_img,recon_flip_pose_flip_pose_img = model(tensor_normal,tensor_normal,tensor_flip)

        recon_normal_exp_flip_pose_img = to_img(recon_normal_exp_flip_pose_img[0])
        recon_flip_exp_normal_pose_img = to_img(recon_flip_exp_normal_pose_img[0])
        recon_normal_exp_normal_exp_img = to_img(recon_normal_exp_normal_pose_img[0])
        recon_normal_pose_normal_pose_img = to_img(recon_normal_pose_normal_pose_img[0])
        recon_flip_pose_flip_pose_img = to_img(recon_flip_pose_flip_pose_img[0])

        if not os.path.exists(save_root):
            os.makedirs(save_root)
        recon_normal_exp_flip_pose_img.save(os.path.join(save_root,img_name + '_' + 'recon_normal_exp_flip_pose.jpg'))
        recon_flip_exp_normal_pose_img.save(os.path.join(save_root,img_name + '_' + 'recon_flip_exp_normal_pose.jpg'))
        recon_normal_exp_normal_exp_img.save(os.path.join(save_root,img_name + '_' + 'recon_normal_exp_normal_exp.jpg'))
        recon_normal_pose_normal_pose_img.save(os.path.join(save_root,img_name + '_' + 'recon_normal_pose_normal_pose.jpg'))
        recon_flip_pose_flip_pose_img.save(os.path.join(save_root,img_name + '_' + 'recon_flip_pose_flip_pose.jpg'))