# import torch
# import torch.utils.data
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import torchvision.models as models
import torchvision.transforms as transforms

# import timm
# from timm.layers import Mlp
import torchvision.models as models
# import torchvision.transforms as transforms
# from main import BatchDataset
#
# # 定义了一个名为Regression的nn.Module子类，用于进行时间预测的回归任务。
# class Regression(nn.Module):
#     """ The header to predict time (regression branch) """
#
#     # num_features表示输入特征的维度，num_classes表示输出的类别数，默认为1
#     def __init__(self, num_features, num_classes=1):
#         super().__init__()#调用父类的初始化函数
#         # 创建一个Mlp实例作为Regression类的成员变量，用于进行特征的非线性映射,num_features: 输入特征的维度。它表示输入特征的大小。
#         # hidden_features: 隐藏层的特征维度。它是输入特征维度的一半，用于定义 MLP 中隐藏层的大小。
#         # out_features: 输出特征的维度。它是输入特征维度的四分之一，用于定义 MLP 的输出大小
#         # self.mlp = Mlp(num_features, hidden_features=num_features//2, out_features=num_features//4, drop=0.5)
#         # 创建一个全连接层作为Regression类的成员变量，用于进行最终的输出,num_features//4: 输入特征的维度。它是 MLP 的输出特征维度的四分之一，用于定义全连接层的输入大小。
#         # num_classes：输出的类别数。对于时间预测任务，该值为1，表示输出的是一个时间值
#         # self.fc = nn.Linear(num_features//4, num_classes)
#
#         self.conv = nn.Sequential(
#             nn.Conv2d(num_features, num_features // 4, kernel_size=3, padding=3 // 2),
#             nn.BatchNorm2d(num_features // 4),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(num_features // 4, num_features // 16, kernel_size=3, padding=3 // 2),
#             nn.BatchNorm2d(num_features // 16),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(num_features // 16, num_features // 32, kernel_size=3, padding=3 // 2),
#         )
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Conv2d(num_features // 32, num_classes, kernel_size=1, bias=True)
#
#         # self.fc = nn.Linear(num_features, num_classes)
#
#     # 定义了Regression类的前向传播函数，接收输入数据x并进行数据的处理和传递
#     def forward(self, x):
#         # x = self.mlp(x)
#         # x = self.fc(x)
#
#         x = x[:, :, None, None]
#
#         x = self.conv(x)
#         x = self.gap(x)
#         x = self.fc(x)
#         x = x.squeeze(-1).squeeze(-1).squeeze(-1)
#         return x
#
#
# class Model(nn.Module):
#     """ A model to predict time"""
#
#     def __init__(self):
#         super().__init__()
#         # 使用timm库的create_model函数创建一个指定名称的模型作为Model类的成员变量，可选是否加载预训练的模型权重
#         self.backbone = models.resnet50(pretrained=True)
#         # 创建一个自适应平均池化层作为Model类的成员变量，用于对每个特征图进行全局平均池化
#         # self.gap = nn.AdaptiveAvgPool2d(1)
#         # self.predictor = Regression(self.backbone.fc.in_features)
#
#     def forward(self, x):
#         # 对输入数据x进行模型的前向计算得到特征图
#         # x = self.backbone.forward_features(x)  # shape: B, D, H, W
#         # 对特征图进行维度转换和自适应平均池化操作，得到固定维度的特征向量,x.permute(0,3,1,2): 对输入的特征图进行维度转换，将原来的通道维度放到第二个位置，将图像宽度和高度的维度分别放到第三和第四个位置。
#         # self.gap: 自适应平均池化层对象，用于对每个特征图进行全局平均池化。
#         # x = self.gap(x).squeeze(dim=(2,3)): 对特征图进行自适应平均池化操作，并使用squeeze函数去除维度为2和3的维度，得到固定维度的特征向量
#         # x = self.gap(x.permute(0,3,1,2)).squeeze(dim=(2,3))
#         # time = self.predictor(x)
#
#         x = self.backbone.conv1(x)
#         x = self.backbone.bn1(x)
#         x = self.backbone.relu(x)
#         x = self.backbone.maxpool(x)
#
#         x = self.backbone.layer1(x)
#         x = self.backbone.layer2(x)
#         x = self.backbone.layer3(x)
#         x = self.backbone.layer4(x)
#
#         x = self.gap(x)
#         x = x.view(x.size(0), -1)
#         time = self.predictor(x)
#         return time
#
# # if __name__ == '__main__':
# #     transform1 = transforms.Compose([
# #         transforms.ToTensor(),
# #         transforms.Resize([224, 224], antialias=True),
# #         # transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
# #         # transforms.RandomRotation(degrees=(0, 180)),
# #     ])
# #     train_dataset = BatchDataset("/home/llj/code/test/data", "/home/llj/code/test/", "train", transform=transform1)
#     # # print(train_dataset[0][0].shape)
#     # model = Model()
#     # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True,
#     #                                            num_workers=2, pin_memory=True)
#     # for i, (inputs, times, filenames) in enumerate(train_loader):
#     #     print(model(inputs))
#
#
# class Model(nn.Module):
#     """ A model to predict time"""
#     def __init__(self, pretrained=True):
#         super().__init__()
#         self.backbone = models.resnet50(pretrained=pretrained)#使用了resnet来提取特征
#         self.gap = nn.AdaptiveAvgPool2d(1)#创建一个自适应平均池化层
#         self.fc = nn.Linear(256, 1) # fc全连接层
#
#     def forward(self, x):
#         #[16, 3, 224, 224]
#         x = self.backbone.conv1(x)#[16, 64, 112, 112]
#         x = self.backbone.bn1(x)
#         x = self.backbone.relu(x)
#         x = self.backbone.maxpool(x)#[16, 64, 56, 56]
#
#         x = self.backbone.layer1(x)#[16, 256, 56, 56]
#         x = self.backbone.layer2(x)#[16, 512, 28, 28]
#         x = self.backbone.layer3(x)#[16, 1024, 14, 14]
#         x = self.backbone.layer4(x)#[16, 2048, 7, 7]
#
#         x = self.gap(x)#[16, 2048, 1, 1]
#         x = x.flatten(1)#[16, 2048]
#         time = self.fc(x)#[16, 1]
#         # time = abs(time)
#         return time
#         # return x

# 线性
# class Model(nn.Module):
#     """ A model to predict time"""
#     def __init__(self):
#         super().__init__()
#         self.fc = nn.Sequential(nn.Flatten(), nn.Linear(256, 1))
#
#     def forward(self, x):
#         time = self.fc(x)
#         return time

#Resnet
# class Model(nn.Module):  #@save
#     def __init__(self, input_channels, num_channels,
#                  use_1x1conv=False, strides=1):
#         super().__init__()
#         self.conv1 = nn.Conv2d(input_channels, num_channels,
#                                kernel_size=3, padding=1, stride=strides)
#         self.conv2 = nn.Conv2d(num_channels, num_channels,
#                                kernel_size=3, padding=1)
#         if use_1x1conv:
#             self.conv3 = nn.Conv2d(input_channels, num_channels,
#                                    kernel_size=1, stride=strides)
#         else:
#             self.conv3 = None
#         self.bn1 = nn.BatchNorm2d(num_channels)
#         self.bn2 = nn.BatchNorm2d(num_channels)
#         self.fc = nn.Linear(256, 1)  # fc全连接层
#
#     def forward(self, X):#[16, 3, 256, 1] 需要(batch_size, input_channels, height, width)
#         # X = X.unsqueeze(1)#[16, 1, 256, 1]
#         Y = F.relu(self.bn1(self.conv1(X)))
#         Y = self.bn2(self.conv2(Y))
#         if self.conv3:
#             X = self.conv3(X)
#         Y += X
#         Y = F.relu(Y)
#         Y = F.relu(self.bn4(self.conv4(Y)))
#         Y = Y.flatten(1)#[16, 256]
#         Y = self.fc(Y)#[16, 1]
#         Y = Y.flatten(0)#[16]
#         return Y

import torch.nn as nn
from torch.nn import functional as F


class RestNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(x + output)


class RestNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = F.relu(self.bn1(output))

        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(extra_x + out)


class RestNet18(nn.Module):
    def __init__(self):
        super(RestNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(RestNetBasicBlock(64, 64, 1),
                                    RestNetBasicBlock(64, 64, 1))

        self.layer2 = nn.Sequential(RestNetDownBlock(64, 128, [2, 1]),
                                    RestNetBasicBlock(128, 128, 1))

        self.layer3 = nn.Sequential(RestNetDownBlock(128, 256, [2, 1]),
                                    RestNetBasicBlock(256, 256, 1))

        self.layer4 = nn.Sequential(RestNetDownBlock(256, 512, [2, 1]),
                                    RestNetBasicBlock(512, 512, 1))

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        out = out.flatten(0)  # [16]
        return out



# class Model(nn.Module):
#     """A model to predict time"""
#     def __init__(self):
#         super(Model, self).__init__()
#         self.resnet = models.resnet18(pretrained=True)
#         self.resnet.fc = nn.Sequential(nn.Flatten(), nn.Linear(256, 1))
#
#     def forward(self, x):
#         x = self.transform_input(x)  # 使用 transforms 进行输入图像处理
#         time = self.resnet(x)
#         return time
#
#     def transform_input(self, x):
#         transform = transforms.Compose([
#             transforms.Grayscale(),  # 将图像转换为灰度图像
#             transforms.ToTensor()  # 将图像转换为张量
#         ])
#         x = transform(x)
#         x = x.repeat(1, 3, 1, 1)  # 将通道数重复为3
#         return x
