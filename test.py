# import torch
# print(torch.__version__)
# print(torch.version.cuda)
# print(torch.backends.cudnn.version())
# print("gpu",torch.cuda.is_available())
# import sys
# print(sys.version)

# -*- "coding: utf-8" -*-
import sys
import os
import time
import logging
import traceback
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

import numpy as np

from main import BatchDataset
from models import ResNet_50
import utils
model = ResNet_50()
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

model.apply(init_weights)
transform1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([224, 224], antialias=True),
    transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
    transforms.RandomRotation(degrees=(0, 180)),
])
# 创建训练集和验证集的数据集对象，包括图像和标签
train_dataset = BatchDataset("/home/llj/code/test/data", "/home/llj/code/test/", "1", transform=transform1)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
# print(type(train_dataset))
print(train_loader)
