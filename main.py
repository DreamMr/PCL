import torch
import utils
import argparse
from torch.utils.tensorboard import SummaryWriter
import os
import torchvision
import tqdm
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from data_process.pfedb import PfeDB
from data_process.expdb import ExpDB
from data_process.posecondb import PoseConDB
from itertools import cycle

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config_file',required=True,type=str)
parser.add_argument('--local_rank',type=int,default=-1)
parser.add_argument('--use_ddp',action='store_true',default=False)

def train(config,pfe_train_loader,exp_train_loader,pose_train_loader,model,logger,step):
    running_dic = None
    count = 0
    #total_num = len(train_loader)
    warm_up = config['warm_up'] if config['warm_up'] is not None else -1
    total_num = len(pfe_train_loader)

    pfe_train_loader = cycle(pfe_train_loader)
    exp_train_loader = cycle(exp_train_loader)
    pose_train_loader = cycle(pose_train_loader)

    exp_loss = 0.
    pose_loss = 0.

    for i in tqdm.tqdm(range(total_num)):

        # pfe train
        data = next(pfe_train_loader)
        data['epoch'] = step
        data['state'] = 'pfe'
        dic = model.optimize_parameters(data)

        # exp train
        if step > warm_up:
            data = next(exp_train_loader)
            data['epoch']= step
            data['state'] = 'exp'
            dic.update(model.optimize_parameters(data))

            data = next(pose_train_loader)
            data['epoch'] = step
            data['state'] = 'pose'
            dic.update(model.optimize_parameters(data))

            exp_loss += dic['exp_loss']
            pose_loss += dic['pose_loss']

        count += 1

        if running_dic == None:
            running_dic = {}
            for k, v in dic.items():
                if k != 'train_print_img':
                    running_dic[k] = v
        else:
            for k, v in dic.items():
                if k != 'train_print_img' and k != 'recon_weight':
                    running_dic[k] += v

        if i % config['print_loss'] == 0:
            txt = 'epoch: {},\t step: {},\t'.format(step, i)
            for k in list(dic.keys()):
                if k != 'train_print_img':
                    txt += ',{}: {},\t'.format(k, dic[k])
            print(txt)

        if config['print_img'] != None and i % config['print_img'] == 0 and 'train_print_img' in dic and dic['train_print_img'] != None:
            print_img = dic['train_print_img']
            grid = torchvision.utils.make_grid(print_img,nrow=1)
            logger.add_image('train_img',grid,global_step=total_num * step + i)

    exp_loss /= count
    pose_loss /= count

    if 'train_loss' in running_dic.keys():
        running_dic['train_loss'] /= count
    if 'train_acc1' in running_dic.keys():
        running_dic['train_acc1'] /= count
    if 'train_acc5' in running_dic.keys():
        running_dic['train_acc5'] /= count

    if 'pose_acc1' in running_dic.keys():
        running_dic['pose_acc1'] /= count
    if 'pose_acc5' in running_dic.keys():
        running_dic['pose_acc5'] /= count
    if 'exp_acc1' in running_dic.keys():
        running_dic['exp_acc1'] /= count
    if 'exp_acc5' in running_dic.keys():
        running_dic['exp_acc5'] /= count

    if 'train_acc1_exp' in running_dic.keys():
        running_dic['train_acc1_exp'] /= count
    if 'train_acc5_exp' in running_dic.keys():
        running_dic['train_acc5_exp'] /= count

    if 'train_acc1_pose' in running_dic.keys():
        running_dic['train_acc1_pose'] /= count
    if 'train_acc5_pose' in running_dic.keys():
        running_dic['train_acc5_pose'] /= count

    if 'train_acc1_flip' in running_dic.keys():
        running_dic['train_acc1_flip'] /= count
    if 'train_acc5_flip' in running_dic.keys():
        running_dic['train_acc5_flip'] /= count

    for k, v in running_dic.items():
        logger.add_scalar(k, v, global_step=step)

    return exp_loss,pose_loss,running_dic['exp_acc1'],running_dic['pose_acc1']

def eval(config,val_loader,model,logger,step):
    running_dic = None
    count = 0
    total_num = len(val_loader)

    for i, data in tqdm.tqdm(enumerate(val_loader)):
        dic = model.eval(data)
        count += 1

        if running_dic == None:
            running_dic = {}
            for k, v in dic.items():
                if k != 'eval_print_img':
                    running_dic[k] = v
        else:
            for k, v in dic.items():
                if k != 'eval_print_img':
                    running_dic[k] += v

        if i % config['print_loss'] == 0:
            txt = 'epoch: {},\t step: {},\t'.format(step, i)
            for k in list(dic.keys()):
                if k != 'eval_print_img':
                    txt += ',{}: {},\t'.format(k, dic[k])
            print(txt)

        if config['print_img'] != None and i % config['print_img'] == 0:
            print_img = dic['eval_print_img']
            grid = torchvision.utils.make_grid(print_img,nrow=1)
            logger.add_image('test_img',grid,global_step=total_num * step + i)

    running_dic['eval_loss'] /= count
    for k, v in running_dic.items():
        logger.add_scalar(k, v, global_step=step)

    return running_dic['eval_loss']

def linear_eval(config,train_loader,val_loader,model,logger):
    count = 0

    linear_classifier = torch.nn.Linear(in_features=config['linear_dim'],out_features=config['classes_num']).cuda()
    linear_classifier.weight.data.normal_(mean=0.0,std=0.01)
    linear_classifier.bias.data.zero_()
    optimizer = torch.optim.Adam(linear_classifier.parameters(),lr=config['linear_lr'])
    lr_schduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config['eval_epochs'])
    criterizer = torch.nn.CrossEntropyLoss().cuda()

    best_knn_acc = 0.
    best_linear_acc = 0.
    knn_not_use = False
    for eval_step in range(config['eval_epochs']):
        train_count = 0
        train_acc_count = 0
        train_loss = 0.
        train_fea_list = []
        train_label_list = []
        for i,data in tqdm.tqdm(enumerate(train_loader)):
            label = data['label'].cuda()
            fea = model.linear_eval(data)
            count += 1

            pred = linear_classifier(fea)

            loss = criterizer(pred,label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_acc_count += (torch.max(pred,dim=1)[1] == label).sum()
            train_count += label.size(0)
            train_loss += loss.item()

            ################### knn #######################
            if knn_not_use:
                b = label.size(0)
                fea_array = fea.detach().cpu().numpy()
                for j in range(b):
                    train_fea_list.append(fea_array[j])
                    train_label_list.append(label[j].item())

        train_acc = train_acc_count / train_count
        train_loss = train_loss / train_count
        lr_schduler.step()

        test_acc_count = 0
        test_knn_acc=0
        test_count = 0
        test_loss = 0.
        linear_classifier.eval()

        test_fea_list = []
        test_label_list = []
        for i,data in tqdm.tqdm(enumerate(val_loader)):
            label = data['label'].cuda()

            fea = model.linear_eval(data)

            with torch.no_grad():
                pred = linear_classifier(fea)

            ################### knn #######################
            if knn_not_use:
                b = label.size(0)
                fea_array = fea.detach().cpu().numpy()
                for j in range(b):
                    test_fea_list.append(fea_array[j])
                    test_label_list.append(label[j].item())

            loss = criterizer(pred,label)
            test_acc_count += (torch.max(pred,dim=1)[1] == label).sum()
            test_count += label.size(0)
            test_loss += loss.item()
        test_linear_acc = test_acc_count / test_count
        test_linear_loss = test_loss / test_count

        ############ knn #############
        if knn_not_use:
            print('knn')
            train_fea = np.array(train_fea_list)
            test_fea = np.array(test_fea_list)

            neigh = KNeighborsClassifier(n_neighbors=5)
            neigh = neigh.fit(train_fea, train_label_list)

            test_pred_list = neigh.predict(test_fea)
            test_count = len(test_label_list)

            test_label_array = np.array(test_label_list)
            test_pred_array = np.array(test_pred_list)
            test_knn_acc_count = (test_pred_array == test_label_array).sum()
            test_knn_acc = test_knn_acc_count / test_count
            best_knn_acc = test_knn_acc
            knn_not_use = False

        if eval_step % config['print_loss'] == 0:
            txt = 'eval step: {},\t linear train acc: {},\t linear train loss: {},\t' \
                  'eval linear acc: {},\t eval linear loss: {},\t knn acc: {}'.format(
                eval_step,train_acc,train_loss,test_linear_acc,test_linear_loss,test_knn_acc
            )
            print(txt)

        logger.add_scalar('linear_train_acc',train_acc,eval_step)
        logger.add_scalar('linear_train_loss',train_loss,eval_step)
        logger.add_scalar('linear_eval_acc', test_linear_acc, eval_step)
        logger.add_scalar('linear_eval_loss', test_linear_loss, eval_step)
        #logger.add_scalar('knn_eval_acc', test_knn_acc, eval_step)

        #best_linear_acc = max(best_linear_acc,test_linear_acc)
        if best_linear_acc < test_linear_acc:
            best_linear_acc = test_linear_acc
            model_state = model.model.state_dict()
            linear_state = linear_classifier.state_dict()
            model_state['linear_classifier.weight'] = linear_state['weight']
            model_state['linear_classifier.bias'] = linear_state['bias']
            utils.save_checkpoint({
                'epoch': eval_step + 1,
                'state_dict': model_state,
        }, config)

    return best_linear_acc,best_knn_acc

def main(config,logger):
    model = utils.create_model(config)
    #train_dataset = utils.create_dataset(config,'train')
    #val_dataset = utils.create_dataset(config,'val')

    if config['eval']:
        test_dataset = utils.create_dataset(config,'test')
        train_dataset = utils.create_dataset(config,'train')

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            num_workers=config['num_threads'],
            pin_memory=True,
            drop_last=False,
            shuffle=False
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            num_workers=config['num_threads'],
            pin_memory=True,
            drop_last=False,
            shuffle=False
        )
        linear_acc,knn_acc = linear_eval(config,train_loader,test_loader,model,logger)
        print('config name: {},\t test linear acc is : {},\t knn acc is : {}\t'.format(config['experiment_name'],linear_acc,knn_acc))
        exit(0)

    pfe_train_dataset = PfeDB(config,phase='train')
    exp_train_dataset = ExpDB(config,phase='train')
    pose_train_dataset= PoseConDB(config,phase='train')

    pfe_train_loader = torch.utils.data.DataLoader(
        pfe_train_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_threads'],
        pin_memory=True,
        drop_last=True,
        shuffle=True
    )
    exp_train_loader = torch.utils.data.DataLoader(
        exp_train_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_threads'],
        pin_memory=True,
        drop_last=True,
        shuffle=True
    )
    pose_train_loader = torch.utils.data.DataLoader(
        pose_train_dataset,
        batch_size=10,
        num_workers=config['num_threads'],
        pin_memory=True,
        drop_last=True,
        shuffle=True
    )

    lr_schduler = torch.optim.lr_scheduler.CosineAnnealingLR(model.optimizer, T_max=config['epochs'])
    best_metric = None

    if config['use_dwa']:
        avg_cost = np.zeros([config['epochs'],2],dtype=np.float32)
        dwa_t = config['dwa_T']
        dwa_start_epoch = config['warm_up'] + 3 if config['warm_up'] !=-1 else 2

    for step in range(config['start_epochs'],config['epochs']+1):
        
        if config['use_dwa']:
            if step > dwa_start_epoch:
                exp_w = avg_cost[step - 1,0] / avg_cost[step-2,0]
                pose_w = avg_cost[step-1,1] / avg_cost[step-2,1]
                model.exp_weight = 2 * np.exp(exp_w / dwa_t) / (np.exp(exp_w / dwa_t) + np.exp(pose_w / dwa_t))
                model.pose_weight = 2 * np.exp(pose_w / dwa_t) / (np.exp(exp_w / dwa_t) + np.exp(pose_w / dwa_t))

                logger.add_scalar('exp_weight', model.exp_weight, step)
                logger.add_scalar('pose_weight',model.pose_weight,step)

        exp_loss,pose_loss,exp_acc,pose_acc = train(config,pfe_train_loader,exp_train_loader,pose_train_loader,model,logger,step)
        print(exp_loss,pose_loss)
        logger.add_scalar('exp_loss_total', exp_loss, step)
        logger.add_scalar('pose_loss_total',pose_loss,step)
        if config['use_dwa']:
            avg_cost[step,0] = exp_loss
            avg_cost[step,1] = pose_loss
        #metric = eval(config,val_loader,model,logger,step)
        lr_schduler.step()

        #flag,cur_best = model.metric_better(metric,best_metric)
        if step % config['save_epoch'] == 0:
            #best_metric = cur_best
            utils.save_checkpoint({
                'epoch': step + 1,
                'state_dict': model.model.state_dict(),
            },config)

if __name__ == '__main__':
    opt = parser.parse_args()

    config = utils.read_config(opt.config_file)
    utils.init(config,opt.local_rank,opt.use_ddp)
    logger = SummaryWriter(log_dir=os.path.join(config['log_path'], config['experiment_name']),
                           comment=config['experiment_name'])

    main(config, logger)

    logger.close()