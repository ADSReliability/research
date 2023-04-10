"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from networks import AdaINGen, MsImageDis, VAEGen
from utils import weights_init, get_model_list, vgg_preprocess, load_vgg16, get_scheduler
from torch.autograd import Variable
import torch
import torch.nn as nn
import os


class MUNIT(nn.Module):
    def __init__(self, hyperparameters):
        super(MUNIT, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        # 生成模型a，即由数据集A到数据集B的映射
        self.gen_a = AdaINGen(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
        # 生成网络模型b, 即由数据集B到数据集A的映射
        self.gen_b = AdaINGen(hyperparameters['input_dim_b'], hyperparameters['gen'])  # auto-encoder for domain b
        # 鉴别模型a，鉴别生成的图像，是否和数据集A的分布一致
        self.dis_a = MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator for domain a
        # 鉴别模型b，鉴别生成的图像，是否和数据集B的分布一致
        self.dis_b = MsImageDis(hyperparameters['input_dim_b'], hyperparameters['dis'])  # discriminator for domain b
        # 使用正则化的方式
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        # style 输出的特征码维度
        self.style_dim = hyperparameters['gen']['style_dim']