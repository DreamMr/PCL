import torch
import utils
import argparse
from torch.utils.tensorboard import SummaryWriter
import os
import torchvision
import tqdm
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pytorch_warmup as warmup
from sklearn.metrics import f1_score
import torch.nn.functional as F
import torch.nn as nn
import math

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config_file',required=True,type=str)
parser.add_argument('--local_rank',type=int,default=-1)
parser.add_argument('--use_ddp',action='store_true',default=False)

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

class FaceCycleBackboneSwapCLR(torch.nn.Module):
    def __init__(self):
        super(FaceCycleBackboneSwapCLR, self).__init__()

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

        self.layer3_2 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1, bias=True),  # stride 2 for 128x128
                                      nn.LeakyReLU(negative_slope=0.1),
                                      nn.Conv2d(256, 128, 3, 1, 1, bias=True),
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
        self.linear_classifier = torch.nn.Sequential(torch.nn.BatchNorm1d(config['linear_dim']),nn.Linear(6272, 3)).cuda()

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
        
        red = self.linear_classifier(out_3)
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
        return red

def eval(config,val_loader,model):
    cos_dis = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    label_list = []
    pred_list = []
    prob = []
    theta = config['theta']

    for i, data in tqdm.tqdm(enumerate(val_loader)):
        label = data['label']

        fea1, fea2 = model.linear_eval_id(data)
        pred = cos_dis(fea1, fea2)
        pred_bool = torch.zeros_like(pred)

        pred_bool[pred >= theta] = 1.
        pred_bool[pred<theta] = -1.
        pred_list.extend(pred_bool.tolist())
        prob.extend(pred.tolist())
        label_list.extend(label.tolist())

    acc_count = 0
    for i in range(len(label_list)):
        print('index: {},\t pred: {},\t prob: {},\t label: {}'.format(i + 1, pred_list[i], prob[i], label_list[i]))
        if pred_list[i] == label_list[i]:
            acc_count += 1

    linear_acc = acc_count / len(label_list)
    return linear_acc

def linear_eval(config,train_loader,val_loader,model,logger):
    best_linear_acc = 100.
    #theta = config['theta']
    if config['pose'] == 'all':
        config['out_dim']=3
    #linear_classifier = torch.nn.Linear(in_features=config['linear_dim'],out_features=config['out_dim']).cuda()
    #linear_classifier = FaceCycleBackboneSwapCLR().cuda()
    linear_classifier = torch.nn.Sequential(torch.nn.BatchNorm1d(num_features=config['linear_dim']),torch.nn.Linear(in_features=config['linear_dim'],out_features=config['out_dim'])).cuda()
    #linear_classifier.weight.data.normal_(mean=0.0,std=0.01)
    #linear_classifier.bias.data.zero_()
    optimizer = torch.optim.AdamW(linear_classifier.parameters(),lr=config['linear_lr'])
    #model = torchvision.models.resnet18(pretrained=False).cuda()
    #optimizer = torch.optim.Adam(model.model.parameters(), lr=config['linear_lr'])
    #optimizer = torch.optim.Adam(model.model.parameters(),lr=config['linear_lr'])
    lr_schduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['eval_epochs'])
    #criterizer = torch.nn.CosineEmbeddingLoss(margin=theta).cuda()
    #criterizer = torch.nn.MSELoss().cuda()
    criterizer = torch.nn.L1Loss()

    if config['linear_eval']:
        for eval_step in range(config['eval_epochs']):
            train_count = 0.
            train_loss = 0.
            cos_dis = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            label_list = []
            pred_list = []
            prob = []
            linear_classifier.train()
            model.model.train()
            train_loss_list = []
            for i, data in tqdm.tqdm(enumerate(train_loader)):
                label = data['label'].float().cuda()
                #print(label)
                #print(label.size())
                fea = model.linear_forward(data)
                pred = linear_classifier(fea)
                #print(fea1.size())
                loss = criterizer(pred,label)
                #print(loss.item())
                #print(loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #train_loss += loss.item()
                train_loss_list.append(loss.item())

            #avg_train_loss = train_loss / train_count
            #acc_count = 0.
            #lr_schduler.step()
            #for i in range(len(label_list)):
            #    if pred_list[i] == label_list[i]:
            #        acc_count += 1
            #train_linear_acc = acc_count / len(label_list)

            linear_classifier.eval()
            model.model.eval()
            eval_loss_list = []
            eval_pitch_loss_list = []
            eval_yaw_loss_list = []
            eval_roll_loss_list = []
            for i,data in tqdm.tqdm(enumerate(val_loader)):
                label = data['label'].float().cuda()

                #fea1,fea2 = model.linear_forward_id(data)

                with torch.no_grad():
                    #fea = data['img_normal'].cuda()
                    fea = model.linear_forward(data)
                    #fea1 = F.normalize(fea1,dim=1)
                    #fea2 = F.normalize(fea2,dim=1)
                    #fea1 = model(data['img_normal1'].cuda())
                    #fea2 = model(data['img_normal2'].cuda())
                    pred = linear_classifier(fea)
                    
                    #label_angle = label * 180 / np.pi
                    #fea = fea * 180 / np.pi
                    loss = criterizer(pred,label)
                    eval_loss_list.append(loss.item())
                    
                    loss_pitch = criterizer(pred[:,0],label[:,0])
                    loss_yaw = criterizer(pred[:,1],label[:,1])
                    loss_roll = criterizer(pred[:,2],label[:,2])
                    eval_pitch_loss_list.append(loss_pitch.item())
                    eval_yaw_loss_list.append(loss_yaw.item())
                    eval_roll_loss_list.append(loss_roll.item())

            avg_train_loss = np.mean(train_loss_list)
            avg_eval_loss = np.mean(eval_loss_list)
            avg_eval_pitch_loss = np.mean(eval_pitch_loss_list)
            avg_eval_yaw_loss = np.mean(eval_yaw_loss_list)
            avg_eval_roll_loss = np.mean(eval_roll_loss_list)
            txt = 'eval step: {},\t train loss: {},\t eval loss: {},\t pitch loss: {},\t yaw loss: {},\t roll loss: {}'.format(eval_step,avg_train_loss,avg_eval_loss,avg_eval_pitch_loss,avg_eval_yaw_loss,avg_eval_roll_loss)
            print(txt)
            #logger.add_scalar('linear_train_acc_id',train_linear_acc,eval_step)
            logger.add_scalar('linear_train_loss_pose',avg_train_loss,eval_step)
            #logger.add_scalar('linear_eval_acc_id',eval_acc,eval_step)
            logger.add_scalar('linear_eval_loss_pose',avg_eval_loss,eval_step)

            if avg_eval_loss < best_linear_acc:
                best_linear_acc = avg_eval_loss
                model_state = model.model.state_dict()
                linear_state = linear_classifier.state_dict()
                for k in list(linear_state.keys()):
                    model_state[k] = linear_state[k]
                utils.save_checkpoint({
                    'epoch': eval_step + 1,
                    'state_dict': model_state,
                }, config)
    else:
        best_linear_acc = eval(config,val_loader,model)
    return best_linear_acc

def main(config,logger):
    model = utils.create_model(config)

    train_dataset = utils.create_dataset(config,'train')
    test_dataset = utils.create_dataset(config,'test')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_threads'],
        pin_memory=True,
        drop_last=False,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_threads'],
        pin_memory=True,
        drop_last=False,
        shuffle=True
    )
    linear_acc = linear_eval(config,train_loader,test_loader,model,logger)
    print('test linear acc is : {}'.format(linear_acc))
    exit(0)


if __name__ == '__main__':
    opt = parser.parse_args()

    config = utils.read_config(opt.config_file)
    utils.init(config,opt.local_rank,opt.use_ddp)
    logger = SummaryWriter(log_dir=os.path.join(config['log_path'], config['experiment_name']),
                           comment=config['experiment_name'])

    main(config, logger)

    logger.close()
