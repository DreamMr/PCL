import yaml
import os
import torch
from torch.backends import cudnn
import random
import importlib
from Models.BaseModel import BaseModel
import numpy as np
import data_process
import torchvision
import time
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

def read_config(yaml_path):
    with open(yaml_path,'r') as imf:
        config = yaml.load(imf.read())
    return config

def save_checkpoint(state,config):
    expr_dir = os.path.join(config['checkpoint_dir'],config['experiment_name'])

    if not os.path.exists(expr_dir):
        os.makedirs(expr_dir)
    epoch = state['epoch']
    save_dir = os.path.join(expr_dir,str(epoch)+'.pth')
    torch.save(state,save_dir)

def set_seed(seed, cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def init(config,local_rank,use_ddp):
    # cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    # cudnn.deterministic = True
    # torch.manual_seed(config['seed'])  # 为CPU设置随机种子
    # torch.cuda.manual_seed(config['seed'])  # 为当前GPU设置随机种子
    # torch.cuda.manual_seed_all(config['seed'])  # 为所有GPU设置随机种子
    # random.seed(config['seed'])
    config['local_rank'] = local_rank
    config['use_ddp'] = use_ddp
    set_seed(config['seed'])

    os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu_ids']

    if config['use_ddp']:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl')

    print_options(config)

def print_options(config):
    expr_dir = os.path.join(config['checkpoint_dir'], config['experiment_name'])

    if not os.path.exists(expr_dir) and ((config['local_rank'] == 0 and config['use_ddp']) or not config['use_ddp']):
        os.makedirs(expr_dir)

    message = '--------------------Options----------------------\n'
    for k in list(config.keys()):
        val = config[k]
        comment = str(k)+':\t'
        if val == None:
            comment += 'None\n'
        else:
            comment += str(val) + '\n'
        message += comment
    message += '--------------------End----------------------\n'

    phase = 'train' if not config['eval'] else 'val'
    file_name = '{}_{}_opt.txt'.format(phase,get_time())
    path = os.path.join(config['checkpoint_dir'],config['experiment_name'],file_name)
    with open(path,'w') as imf:
        imf.write(message)

def get_time():
    return str(time.strftime("%Y_%m_%d_%H_%M_%S",time.localtime()))

def create_model(config):
    model_name = 'Models.' + config['model_name']
    modellib = importlib.import_module(model_name)
    model = None
    target_model_name = config['model_name']
    for name,cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() and issubclass(cls,BaseModel):
            model = cls

    if model == None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (
        model_name, target_model_name))
        exit(0)

    instance = model(config)
    return instance

def create_dataset(config,phase=None):
    '''

    :param config:
    :param phase: 'train,val,test' look at config
    :return:
    '''
    dataset_name = 'data_process.' + config['dataset_name']
    modellib = importlib.import_module(dataset_name)
    dataset = None
    target_dataset_name = config['dataset_name']
    for name,cls in modellib.__dict__.items():
        if name.lower() == target_dataset_name:
            dataset = cls

    if dataset == None:
        print("In %s.py, there should has class name that matches %s in lowercase." % (
            dataset_name, target_dataset_name))
        exit(0)

    instance = dataset(config,phase)
    return instance

def img2label(colormap,config):
    cm2lbl = np.zeros(config['img_size'] ** 3,dtype='int64')
    for i, cm in enumerate(colormap):
        cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i
    return cm2lbl

def iou(pred, target,n_class):
    ious = []
    for cls in range(n_class):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.int().sum() + target_inds.int().sum() - intersection
        if union == 0:
            ious.append(0)  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / max(float(union), 1))
        # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))
    return ious

def cal_matrics(pred,label):
    p = torch.flatten(pred)
    l = torch.flatten(label)
    p = p.detach().cpu().numpy().tolist()
    l = l.detach().cpu().numpy().tolist()
    tn1, fp1, fn1, tp1 = confusion_matrix(l, p, labels=[0, 1]).flatten()
    f1 = (2 * tp1) / (2 * tp1 + fn1 + fp1)
    iou = tp1 / (fn1+fp1+tp1)
    return f1,iou

def cal_f1_score(pred,label):
    '''

    :param pred: [1,w,h]
    :param label: [1,w,h]
    :return: float f1 score
    '''
    p = torch.flatten(pred)
    l = torch.flatten(label)
    p = p.detach().cpu().numpy().tolist()
    l = l.detach().cpu().numpy().tolist()
    #tn1, fp1, fn1, tp1 = confusion_matrix(l, p, labels=[0, 1]).flatten()
    #f1 = (2 * tp1) / (2 * tp1 + fn1 + fp1)
    #return f1
    f1 = f1_score(l,p,average=None)
    if len(f1) >= 2:
        return f1[1]
    else:
        return f1[0]

def save_img(img,img_name,config):
    dir_path = os.path.join(config['save_img_dir'],config['experiment_name'])
    file_path = os.path.join(dir_path, img_name)

    preffix,suffix = os.path.split(file_path)
    if not os.path.exists(preffix):
        os.makedirs(preffix)

    torchvision.utils.save_image(img,file_path)

def accuracy(output,target,topk=(1,)):

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _,pred = output.topk(maxk,1,True,True)
        pred = pred.t()
        correct = pred.eq(target.view(1,-1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0,keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())

        return res

# if __name__ == '__main__':
#     yaml_path = './configs/init.yaml'
#     config = read_config(yaml_path)
#
#     model = create_dataset(config)
#     debug = 0