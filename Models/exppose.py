import numpy as np
import torch
from Models.networks import *
from Models.BaseModel import BaseModel
import utils
import torchvision.models as models
import time
import math
from torch.cuda.amp import GradScaler, autocast


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

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))
        #diff_loss = torch.mean((input1_l2 * input2_l2).sum(dim=1).pow(2))

        return diff_loss

class ExpPose(BaseModel):
    def __init__(self,config):
        BaseModel.__init__(self,config)

        self.temperature = config['T']
        self.batch_size = config['batch_size']
        self.neg_alpha = config['neg_alpha']
        self.pose_alpha = config['pose_alpha']
        self.warm_up_epoch = config['warm_up'] if config['warm_up'] is not None else -1

        self.exp_grad = 0.
        self.pose_grad = 0.
        self.cos = torch.nn.CosineSimilarity(dim=1)

        if not config['eval'] and not config['t_sne']:
            self.model = ExpPoseModel()
        else:
            #face_cycle_backbone = FaceCycleBackbone()
            #face_cycle_backbone = SimCLRNetwork()
            face_cycle_backbone = FaceCycleBackbone()
            #face_cycle_backbone=FaceCycleBackboneSingle()
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
            state_dict = torch.load(config['load_model'])['state_dict']
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
                    elif k.startswith('module.exp_fc') or k.startswith('module.pose_fc'):
                        state_dict[k[len("module."):]] = state_dict[k]
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
                    if k.startswith('encoder.pose.'):
                        #state_dict['fc.' + k[len("encoder.exp.")]] = state_dict[k]
                        state_dict['fc.' + k[len("encoder.pose.")]] = state_dict[k]
                    if k.startswith('encoder.'):
                        state_dict[k[len("encoder."):]] = state_dict[k]
                    del state_dict[k]

            #self.model = self.model.cuda()
            #self.model = torch.nn.DataParallel(self.model).cuda()
            #print(state_dict.keys())
            msg = self.model.load_state_dict(state_dict,strict=False)
            #print(msg)
            assert set(msg.missing_keys) == set()
            print('load model !')
        else:
            self.model = torch.nn.DataParallel(self.model).cuda()

        self.criterion = nn.CrossEntropyLoss().cuda()
        self.recon_criterion = nn.L1Loss().cuda()
        self.diff_loss = DiffLoss().cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=config['lr'],weight_decay=config['wd'])
        self.scaler = GradScaler()

        self.exp_weight = 1.
        self.pose_weight = 1.

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


    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda()

        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features,features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        labels = labels[~mask].view(labels.shape[0], -1) # ~mask 取反
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1) # 去除了自己
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        logits = logits / self.temperature
        return logits, labels

    # todo pose contrastive loss
    def optimize_parameters(self,data):
        self.model.train()

        #img_normal = data['img_normal'].cuda()
        #img_flip = data['img_flip'].cuda()
        #img_normal = data['exp_images'][0].cuda()
        #img_flip = data['exp_images'][1].cuda()
        cur_epoch = data['epoch']
        state = data['state']

        log_dic = {}
        if cur_epoch > self.warm_up_epoch:
            with autocast():
                if state == 'pfe':
                    img_normal = data['img_normal'].cuda()
                    img_flip = data['img_flip'].cuda()
                    normal_exp_fea, normal_pose_fea, flip_exp_fea, flip_pose_fea, recon_normal_exp_flip_pose_img, recon_flip_exp_normal_pose_img, recon_normal_exp_normal_pose_img, recon_flip_exp_flip_psoe_img = self.forward(
                        data, recon_only=True)

                    recon_normal_loss = self.recon_criterion(recon_flip_exp_normal_pose_img, img_normal)  # || s-s'||
                    # recon_normal_loss = 0.
                    recon_flip_loss = self.recon_criterion(recon_normal_exp_flip_pose_img, img_flip)  # ||f-f'||
                    # recon_flip_loss = 0.
                    recon_orin_loss = self.recon_criterion(recon_normal_exp_normal_pose_img, img_normal)  # ||s-s''||
                    # recon_orin_loss = 0.
                    recon_flip_ori_loss = self.recon_criterion(recon_flip_exp_flip_psoe_img, img_flip)

                    # recon_weight = self.get_lambda(cur_epoch)
                    recon_weight = 1.
                    diff_loss = self.diff_loss(normal_exp_fea, normal_pose_fea) + self.diff_loss(flip_exp_fea,
                                                                                                 flip_pose_fea)

                    loss_pfe = recon_weight * (recon_normal_loss + recon_flip_loss + recon_orin_loss + recon_flip_ori_loss)
                    loss = diff_loss + loss_pfe

                    print_img = torch.cat([img_normal[:2], img_flip[:2], recon_flip_exp_normal_pose_img[:2],
                                           recon_normal_exp_flip_pose_img[:2], recon_normal_exp_normal_pose_img[:2]],
                                          dim=3)
                    log_dic = {
                        'train_diff_loss': diff_loss.item(),
                        'train_recon_normal_loss':recon_normal_loss.item(),
                        'train_recon_flip_loss':recon_flip_loss.item(),
                        'train_recon_orin_loss':recon_orin_loss.item(),
                        'train_recon_flip_orin_loss':recon_flip_ori_loss.item(),
                        }
                    log_dic['train_print_img'] = print_img

                elif state =='exp':
                    exp_fea_fc = self.forward(data)
                    exp_logits, exp_labels = self.neg_inter_info_nce_loss(exp_fea_fc)
                    loss = self.criterion(exp_logits, exp_labels)

                    exp_acc1, exp_acc5 = utils.accuracy(exp_logits, exp_labels, (1, 5))
                    log_dic = {
                        'exp_acc1':exp_acc1,'exp_acc5':exp_acc5,
                        'exp_loss':loss.item()
                    }
                    
                    # weight
                    loss = loss * self.exp_weight

                elif state =='pose':
                    pose_fea_fc,pose_neg_fea_fc = self.forward(data)
                    pose_logits,pose_labels = self.neg_inter_info_nce_loss_pose(pose_fea_fc,pose_neg_fea_fc)
                    loss = self.criterion(pose_logits,pose_labels)

                    pose_acc1,pose_acc5=utils.accuracy(pose_logits,pose_labels,(1,5))
                    log_dic={
                        'pose_acc1':pose_acc1,'pose_acc5':pose_acc5,
                        'pose_loss':loss.item()
                    }

                    # weight
                    loss = loss * self.pose_weight
        else:
            with autocast():
                img_normal = data['img_normal'].cuda()
                img_flip = data['img_flip'].cuda()
                # exp_fea, pose_fea, recon_normal_exp_flip_pose_img, recon_flip_exp_normal_pose_img,recon_normal_exp_normal_pose_img,normal_exp_fea_fc,normal_pose_fea_fc,flip_exp_fea_fc,flip_pose_fea_fc = self.forward(data)
                normal_exp_fea,normal_pose_fea,flip_exp_fea,flip_pose_fea, recon_normal_exp_flip_pose_img,recon_flip_exp_normal_pose_img,recon_normal_exp_normal_pose_img,recon_flip_exp_flip_psoe_img = self.forward(data,recon_only=True)

                recon_normal_loss = self.recon_criterion(recon_flip_exp_normal_pose_img, img_normal)  # || s-s'||
                # recon_normal_loss = 0.
                recon_flip_loss = self.recon_criterion(recon_normal_exp_flip_pose_img, img_flip)  # ||f-f'||
                # recon_flip_loss = 0.
                recon_orin_loss = self.recon_criterion(recon_normal_exp_normal_pose_img, img_normal)  # ||s-s''||
                # recon_orin_loss = 0.
                recon_flip_ori_loss = self.recon_criterion(recon_flip_exp_flip_psoe_img, img_flip)
                recon_flip_ori_loss = 0.

                # recon_weight = self.get_lambda(cur_epoch)
                recon_weight = 1.
                diff_loss = self.diff_loss(normal_exp_fea, normal_pose_fea) + self.diff_loss(flip_exp_fea,
                                                                                             flip_pose_fea)

                loss = recon_weight * (recon_normal_loss + recon_flip_loss + recon_orin_loss + recon_flip_ori_loss)

            log_dic = {'train_loss': loss.item(),
                       'train_diff_loss': diff_loss.item(),
                       'train_recon_normal_loss': recon_normal_loss.item(),
                       'train_recon_flip_loss': recon_flip_loss.item(),
                       'train_recon_orin_loss': recon_orin_loss.item(),
                       #'train_recon_flip_orin_loss': recon_flip_ori_loss.item(),
                       }
            print_img = torch.cat(
                [img_normal[:2], img_flip[:2], recon_flip_exp_normal_pose_img[:2], recon_normal_exp_flip_pose_img[:2],
                 recon_normal_exp_normal_pose_img[:2]], dim=3)
            log_dic['train_print_img'] = print_img

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return log_dic

    # todo pose contrastive loss
    def info_nec_loss_pose(self,features,pose_nag_fea):
        b,dim = features.size()

        neg_mask = np.random.beta(self.neg_alpha, self.neg_alpha,
                                  size=(pose_nag_fea.shape[0]))
        time_alpha_finish = time.time()
        # print('cost alpha time: {}'.format(time_alpha_finish - time_alpha))
        if isinstance(neg_mask, np.ndarray):
            neg_mask = torch.from_numpy(neg_mask).float().cuda()
            neg_mask = neg_mask.unsqueeze(dim=1)
        indices = torch.randperm(pose_nag_fea.shape[0])
        pose_nag_fea = pose_nag_fea * neg_mask + (1 - neg_mask) * pose_nag_fea[indices,:]

        features = F.normalize(features,dim=1)
        q,k = features.chunk(2,dim=0)
        fea_neg_tensor = F.normalize(pose_nag_fea,dim=1)

        pos = torch.cat(
            [torch.einsum('nc,nc->n',[q,k]).unsqueeze(-1),
             torch.einsum('nc,nc->n',[k,q]).unsqueeze(-1)],dim=0)

        ## todo repeat
        #fea_neg_tensor = fea_neg_tensor.repeat(2,1)
        fea_neg_tensor = fea_neg_tensor.transpose(1,0)
        neg = torch.mm(features,fea_neg_tensor).view(b,-1)

        logits = torch.cat([pos,neg],dim=1)
        labels = torch.zeros(logits.shape[0],dtype=torch.long).cuda()

        logits = logits / self.temperature

        return logits, labels

    def neg_inter_info_nce_loss(self, features,pose_flag=False,flip_pose=None):

        # time_start = time.time()

        b, dim = features.size()

        labels = torch.cat([torch.arange(b // 2) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        labels_flag = (1 - labels).bool()
        features_expand = features.expand((b, b, dim))  # 512 * 512 * dim
        fea_neg_li = list(features_expand[labels_flag].chunk(b, dim=0))
        fea_neg_tensor = torch.stack(fea_neg_li, dim=0)  # 512 * 510 * dim
        if pose_flag:
            flip_pose = flip_pose.repeat(2,1)
            fea_neg_tensor = torch.cat([fea_neg_tensor,flip_pose.view(b,1,-1)],dim=1)

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

    def neg_inter_info_nce_loss_pose(self,features,neg_pose_fea):
        b,dim=features.size()

        neg_pose_fea_li = list(neg_pose_fea.chunk(b//2,dim=0))
        neg_pose_fea_tensor = torch.stack(neg_pose_fea_li,dim=0)

        neg_mask = np.random.beta(self.neg_alpha, self.neg_alpha,
                                  size=(neg_pose_fea_tensor.shape[0], neg_pose_fea_tensor.shape[1]))
        if isinstance(neg_mask,np.ndarray):
            neg_mask = torch.from_numpy(neg_mask).float().cuda()
            neg_mask = neg_mask.unsqueeze(dim=2)
        indices = torch.randperm(neg_pose_fea_tensor.shape[1])
        neg_pose_fea_tensor = neg_pose_fea_tensor * neg_mask + (1 - neg_mask) * neg_pose_fea_tensor[:,indices]

        features = F.normalize(features,dim=1)
        q,k = features.chunk(2,dim=0)
        neg_pose_fea_tensor = F.normalize(neg_pose_fea_tensor,dim=2)
        neg_pose_fea_tensor = neg_pose_fea_tensor.repeat(2,1,1)

        pos = torch.cat(
            [torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1), torch.einsum('nc,nc->n', [k, q]).unsqueeze(-1)], dim=0)

        neg_pose_fea_tensor = neg_pose_fea_tensor.transpose(2,1)
        neg = torch.bmm(features.view(b,1,-1),neg_pose_fea_tensor).view(b,-1)

        logits = torch.cat([pos,neg],dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        logits = logits / self.temperature

        return logits,labels


    
    def forward(self,data,recon_only=False):
        if data['state'] == 'pfe':
            # exp_images = data['exp_images']
            # exp_image_pose_neg = data['exp_image_pose_neg'].cuda()
            img_normal = (data['img_normal'] + torch.randn_like(data['img_normal'])*4e-1).cuda()
            img_flip = (data['img_flip'] + torch.randn_like(data['img_normal'])*4e-1).cuda()
            #exp_images = torch.cat(exp_images,dim=0).cuda()
            return self.model(normal_img=img_normal,flip_img=img_flip,recon_only=recon_only,state=data['state'])
        elif data['state'] == 'exp':
            exp_images = data['exp_images']
            exp_images = torch.cat(exp_images,dim=0).cuda()
            return self.model(exp_img=exp_images,state=data['state'])
        elif data['state'] == 'pose':
            exp_images = data['exp_images']
            neg_images_li = data['neg_images']
            exp_images = torch.cat(exp_images,dim=0).cuda()
            neg_images = torch.cat(neg_images_li,dim=0).cuda()
            #print(neg_images.size())
            return self.model(exp_img=exp_images,exp_image_pose_neg=neg_images,recon_only=recon_only,state=data['state'])
        
    def linear_forward(self,data):
        img = data['img_normal'].float().cuda()
        fea,exp_fea,_ = self.model(img)
        return fea

    def linear_forward_id(self,data):
        img1 = data['img_normal1'].cuda()
        fea1,exp_fea1,pose_fea1 = self.model(img1)

        img2 = data['img_normal2'].cuda()
        fea2,exp_fea2,pose_fea2 = self.model(img2)

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




