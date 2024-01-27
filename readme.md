# Pose-disentangled Contrastive Learning for Self-supervised Facial Representation

This repository is the Pytorch implementation for our CVPR2023 paper: **Pose-disentangled Contrastive Learning for Self-supervised Facial Representation**.

paper link: [arxiv](https://arxiv.org/abs/2211.13490)

## 0. Contents

1. Requirements
2. Data Preparation
3. Pre-trained Models
4. Training
5. Evaluation

## 1. Requirements

To install requirements:
Python Version: 3.7.9

```
pip install -r requirements.txt
```

## 2. Data Preparation

You need to download the related datasets  and put in the folder which namely dataset.

## 3. Pre-trained Models

You can download our trained models from [Baidu Drive](https://pan.baidu.com/s/10j21PCyhi9cbJqRvH7KDHw) (2qia) and [Google Drive](https://drive.google.com/drive/folders/1wx5PTGDCqDWsjhXimjHqz_7WUwxr54uh?usp=sharing) .

## 4. Training

To train the model in the paper, run this command:

```
python main.py --config_file configs/remote_PCL_vox.yaml
```

## 5. Evaluation

We used the linear evaluation protocol for evaluation.

### 5.1 FER

To evaluate on RAF-DB, run:

```
python main.py --config_file configs/remote_PCL_linear_eval.yaml
```

### 5.2 Pose regression

To trained on 300W-LP and evaluated on AFLW2000, run:

```
python main_pose.py --config_file configs/remote_PCL_linear_eval_pose.yaml
```

### 5.3 Visualization

To visualize on RAF-DB, run:

```
python visualize.py
```



## TODO 

- [ ] Refactor the codes of AU detection and face recognition.

**IF YOU HAVE ANY PROBLEM, PLEASE CONTACT wangwenbin@cug.edu.cn OR COMMIT ISSUES**

