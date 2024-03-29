# import torch
# import torch.utils.data
import torch
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
class Model(nn.Module):  #@save
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.fc = nn.Linear(256, 1)  # fc全连接层

    def forward(self, X):#[16, 3, 256, 1] 需要(batch_size, input_channels, height, width)
        # X = X.unsqueeze(1)#[16, 1, 256, 1]
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        Y = F.relu(Y)
        # Y = F.relu(self.bn4(self.conv4(Y)))
        Y = Y.flatten(1)#[16, 256]
        Y = self.fc(Y)#[16, 1]
        Y = Y.flatten(0)#[16]
        return Y

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
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3)
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

        # self.fc = nn.Linear(512, 1)
        self.fc = nn.Linear(512, 1)

    def softmax(X):
        X_exp = torch.exp(X)
        partition = X_exp.sum(1, keepdim=True)
        return X_exp / partition  # 这里应用了广播机制

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

#Resnet50
# class SELayer(nn.Module):
#     def __init__(self, channel=64, reduction=16):
#         super(SELayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y.expand_as(x)

# class SelfAttention(nn.Module):
#     def __init__(self, in_channels, ratio=8):
#         super(SelfAttention, self).__init__()
#         self.query_conv = nn.Conv2d(in_channels, in_channels // ratio, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels, in_channels // ratio, kernel_size=1)
#         self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#     def forward(self, x):
#         batch_size, channels, width, height = x.size()
#         # print(batch_size, channels, width, height)
#         proj_query = self.query_conv(x).view(batch_size, -1, width * height)#[16, 32, 256]
#         proj_key = self.key_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)#[16, 256, 32]
#         energy = torch.bmm(proj_query, proj_key)#[16, 32, 32]
#         attention = F.softmax(energy, dim=-1)#[16, 32, 32]
#         attention = attention.unsqueeze(2)
#         print(attention.shape)
#         proj_value = self.value_conv(x).view(batch_size, -1, width * height)#[16, 256, 256]
#
#         out = torch.bmm(attention, proj_value)
#         print(out.shape)
#         out = out.view(batch_size, channels, width, height)
#
#         out = self.gamma * out + x
#         return out
class Bottleneck(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(Bottleneck, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, int(outchannel / 4), kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(int(outchannel / 4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(outchannel / 4), int(outchannel / 4), kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(int(outchannel / 4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(outchannel / 4), outchannel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(outchannel),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        y = self.shortcut(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out
class ResNet_50(nn.Module):
    def __init__(self):
        super(ResNet_50, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(Bottleneck, 256, 3, stride=1)
        self.layer2 = self.make_layer(Bottleneck, 512, 4, stride=2)
        self.layer3 = self.make_layer(Bottleneck, 1024, 6, stride=2)
        self.layer4 = self.make_layer(Bottleneck, 2048, 3, stride=2)
        self.fc = nn.Linear(1024, 1)
        # self.se = SELayer()
        # self.attention1 = SelfAttention(256)
        # self.attention2 = SelfAttention(256)

        # **************************

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):# [16, 4, 16, 16]
        out = self.conv1(x)  # [16, 64, 16, 16]
        # out = self.se(out)
        # print(out.shape)
        out = self.layer1(out)#[16, 256, 16, 16]
        # 添加自注意力机制
        # out = self.attention1(out)
        out = self.layer2(out)#[16, 512, 8, 8]
        # 添加自注意力机制
        # out = self.attention2(out)
        out = self.layer3(out)#[16, 1024, 4, 4]
        # out = self.layer4(out)#[16, 2048, 2, 2]
        # out = F.avg_pool2d(out, 7)  # layer2 [16, 1024, 1, 1]
        out = F.avg_pool2d(out, 4)#layer3 [16, 1024, 1, 1]
        # out = F.avg_pool2d(out, 2)# layer4 [16, 2048, 1, 1]
        # print(out.shape)
        out = out.view(out.size(0), -1)#[16, 2048]
        out = self.fc(out)#[16,1]
        out = out.flatten(0)#[16]
        return out





#SSR_Net
class SSRNet(nn.Module):
    def __init__(self, stage_num=[3,3,3,3], image_size=16,
                 class_range=144, lambda_index=1., lambda_delta=1.):
        super(SSRNet, self).__init__()
        self.image_size = image_size
        self.stage_num = stage_num
        self.lambda_index = lambda_index
        self.lambda_delta = lambda_delta
        self.class_range = class_range

        self.stream1_stage4 = nn.Sequential(
            nn.Conv2d(4, 32, 3, 1, 1),  # O = （I - K + 2P）/ S +1 (如16x16的输入，O=(16-3+2)/1+1=16    [16, 32, 16, 16]
            nn.BatchNorm2d(32),  # [16, 32, 16, 16] 归一化输入输出形状相同
            nn.ReLU(),  # [16, 32, 16, 16]，ReLU(x)=max(0,x)，输入输出形状相同
            nn.AvgPool2d(2, 2)  # [16, 32, 8, 8]
        )
        self.stream1_stage3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2, 2)
        )
        self.stream1_stage2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2, 2)
        )
        self.stream1_stage1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.AvgPool2d(2, 2) # paper has this layer, but official codes don't.
        )
        self.stream2_stage4 = nn.Sequential(
            nn.Conv2d(4, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.MaxPool2d(2, 2)
        )
        self.stream2_stage3 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.MaxPool2d(2, 2)
        )
        self.stream2_stage2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.MaxPool2d(2, 2)
        )
        self.stream2_stage1 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            # nn.MaxPool2d(2, 2) # paper has this layer, but official codes don't.
        )

        # fusion block
        self.funsion_block_stream1_stage_4_before_PB = nn.Sequential(
            nn.Conv2d(32, 10, 1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(8, 8)
        )
        self.funsion_block_stream1_stage_4_prediction_block = nn.Sequential(
            nn.Dropout(0.2, ),
            nn.Linear(10, self.stage_num[3]),
            nn.ReLU()
        )

        self.funsion_block_stream1_stage_3_before_PB = nn.Sequential(
            nn.Conv2d(32, 10, 1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(4, 4)
        )
        self.funsion_block_stream1_stage_3_prediction_block = nn.Sequential(
            nn.Dropout(0.2, ),
            nn.Linear(10, self.stage_num[2]),
            nn.ReLU()
        )

        self.funsion_block_stream1_stage_2_before_PB = nn.Sequential(
            nn.Conv2d(32, 10, 1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(2, 2)
        )
        self.funsion_block_stream1_stage_2_prediction_block = nn.Sequential(
            nn.Dropout(0.2, ),
            nn.Linear(10, self.stage_num[1]),
            nn.ReLU()
        )

        self.funsion_block_stream1_stage_1_before_PB = nn.Sequential(
            nn.Conv2d(32, 10, 1, padding=0),
            nn.ReLU(),
            # nn.AvgPool2d(2, 2) # paper has this layer, but official codes don't.
        )
        self.funsion_block_stream1_stage_1_prediction_block = nn.Sequential(
            nn.Dropout(0.2, ),
            nn.Linear(10, self.stage_num[0]),
            nn.ReLU()
        )

        # stream2
        self.funsion_block_stream2_stage_4_before_PB = nn.Sequential(
            nn.Conv2d(16, 10, 1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(8, 8)
        )
        self.funsion_block_stream2_stage_4_prediction_block = nn.Sequential(
            nn.Dropout(0.2, ),
            nn.Linear(10, self.stage_num[3]),
            nn.ReLU()
        )
        self.funsion_block_stream2_stage_3_before_PB = nn.Sequential(
            nn.Conv2d(16, 10, 1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(4, 4)
        )
        self.funsion_block_stream2_stage_3_prediction_block = nn.Sequential(
            nn.Dropout(0.2, ),
            nn.Linear(10, self.stage_num[2]),
            nn.ReLU()
        )

        self.funsion_block_stream2_stage_2_before_PB = nn.Sequential(
            nn.Conv2d(16, 10, 1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.funsion_block_stream2_stage_2_prediction_block = nn.Sequential(
            nn.Dropout(0.2, ),
            nn.Linear(10, self.stage_num[1]),
            nn.ReLU()
        )

        self.funsion_block_stream2_stage_1_before_PB = nn.Sequential(
            nn.Conv2d(16, 10, 1, padding=0),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2) # paper has this layer, but official codes don't.
        )
        self.funsion_block_stream2_stage_1_prediction_block = nn.Sequential(
            nn.Dropout(0.2, ),
            nn.Linear(10, self.stage_num[0]),
            nn.ReLU()
        )

        self.stage4_FC_after_PB = nn.Sequential(
            nn.Linear(self.stage_num[0], 2 * self.stage_num[0]),
            nn.ReLU()
        )
        self.stage4_prob = nn.Sequential(
            nn.Linear(2 * self.stage_num[0], self.stage_num[0]),
            nn.ReLU()
        )
        self.stage4_index_offsets = nn.Sequential(
            nn.Linear(2 * self.stage_num[0], self.stage_num[0]),
            nn.Tanh()
        )
        self.stage4_delta_k = nn.Sequential(
            nn.Linear(10, 1),
            nn.Tanh()
        )

        self.stage3_FC_after_PB = nn.Sequential(
            nn.Linear(self.stage_num[0], 2 * self.stage_num[0]),
            nn.ReLU()
        )
        self.stage3_prob = nn.Sequential(
            nn.Linear(2 * self.stage_num[0], self.stage_num[0]),
            nn.ReLU()
        )
        self.stage3_index_offsets = nn.Sequential(
            nn.Linear(2 * self.stage_num[0], self.stage_num[0]),
            nn.Tanh()
        )
        self.stage3_delta_k = nn.Sequential(
            nn.Linear(10, 1),
            nn.Tanh()
        )

        self.stage2_FC_after_PB = nn.Sequential(
            nn.Linear(self.stage_num[0], 2 * self.stage_num[0]),
            nn.ReLU()
        )
        self.stage2_prob = nn.Sequential(
            nn.Linear(2 * self.stage_num[0], self.stage_num[0]),
            nn.ReLU()
        )
        self.stage2_index_offsets = nn.Sequential(
            nn.Linear(2 * self.stage_num[0], self.stage_num[0]),
            nn.Tanh()
        )
        self.stage2_delta_k = nn.Sequential(
            nn.Linear(10, 1),
            nn.Tanh()
        )

        self.stage1_FC_after_PB = nn.Sequential(
            nn.Linear(self.stage_num[0], 2 * self.stage_num[0]),
            nn.ReLU()
        )
        self.stage1_prob = nn.Sequential(
            nn.Linear(2 * self.stage_num[0], self.stage_num[0]),
            nn.ReLU()
        )
        self.stage1_index_offsets = nn.Sequential(
            nn.Linear(2 * self.stage_num[0], self.stage_num[0]),
            nn.Tanh()
        )
        self.stage1_delta_k = nn.Sequential(
            nn.Linear(10, 1),
            nn.Tanh()
        )
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, image_):  # [16, 4, 16, 16]
        feature_stream1_stage4 = self.stream1_stage4(image_)  # [16, 32, 8, 8]
        feature_stream1_stage3 = self.stream1_stage3(feature_stream1_stage4)  # [16, 32, 8, 8]

        feature_stream1_stage2 = self.stream1_stage2(feature_stream1_stage3)  # [16, 32, 4, 4]

        feature_stream1_stage1 = self.stream1_stage1(feature_stream1_stage2)  # [16, 32, 2, 2]

        feature_stream2_stage4 = self.stream2_stage4(image_)
        feature_stream2_stage3 = self.stream2_stage3(feature_stream2_stage4)  # [16, 16, 8, 8]

        feature_stream2_stage2 = self.stream2_stage2(feature_stream2_stage3)  # [16, 16, 4, 4]

        feature_stream2_stage1 = self.stream2_stage1(feature_stream2_stage2)  # [16, 16, 2, 2]

        feature_stream1_stage4_before_PB = self.funsion_block_stream1_stage_4_before_PB(
            feature_stream1_stage4)  # [16, 10, 1, 1]
        feature_stream1_stage3_before_PB = self.funsion_block_stream1_stage_3_before_PB(
            feature_stream1_stage3)  # [16, 10, 1, 1]
        feature_stream1_stage2_before_PB = self.funsion_block_stream1_stage_2_before_PB(
            feature_stream1_stage2)  # [16, 10, 1, 1]
        feature_stream1_stage1_before_PB = self.funsion_block_stream1_stage_1_before_PB(
            feature_stream1_stage1)  # [16, 10, 2, 2]

        feature_stream2_stage4_before_PB = self.funsion_block_stream2_stage_4_before_PB(
            feature_stream2_stage4)  # [16, 10, 1, 1]
        feature_stream2_stage3_before_PB = self.funsion_block_stream2_stage_3_before_PB(
            feature_stream2_stage3)  # [16, 10, 1, 1]
        feature_stream2_stage2_before_PB = self.funsion_block_stream2_stage_2_before_PB(
            feature_stream2_stage2)  # [16, 10, 1, 1]
        feature_stream2_stage1_before_PB = self.funsion_block_stream2_stage_1_before_PB(
            feature_stream2_stage1)  # [16, 10, 2, 2]

        # △k
        embedding_stream1_stage4_before_PB = feature_stream1_stage4_before_PB.view(
            feature_stream1_stage4_before_PB.size(0), -1)
        embedding_stream1_stage3_before_PB = feature_stream1_stage3_before_PB.view(
            feature_stream1_stage3_before_PB.size(0), -1)  # [16, 10]
        embedding_stream1_stage2_before_PB = feature_stream1_stage2_before_PB.view(
            feature_stream1_stage2_before_PB.size(0), -1)  # [16, 10]
        embedding_stream1_stage1_before_PB = feature_stream1_stage1_before_PB.view(
            feature_stream1_stage1_before_PB.size(0), -1)  # [16, 40]
        embedding_stream2_stage4_before_PB = feature_stream2_stage4_before_PB.view(
            feature_stream2_stage4_before_PB.size(0), -1)
        embedding_stream2_stage3_before_PB = feature_stream2_stage3_before_PB.view(
            feature_stream2_stage3_before_PB.size(0), -1)  # [16, 10]
        embedding_stream2_stage2_before_PB = feature_stream2_stage2_before_PB.view(
            feature_stream2_stage2_before_PB.size(0), -1)  # [16, 10]
        embedding_stream2_stage1_before_PB = feature_stream2_stage1_before_PB.view(
            feature_stream2_stage1_before_PB.size(0), -1)  # [16, 40]
        stage1_delta_k = self.stage1_delta_k(
            torch.mul(embedding_stream1_stage1_before_PB, embedding_stream2_stage1_before_PB))  # [16, 1]
        stage2_delta_k = self.stage2_delta_k(
            torch.mul(embedding_stream1_stage2_before_PB, embedding_stream2_stage2_before_PB))  # [16, 1]
        stage3_delta_k = self.stage3_delta_k(
            torch.mul(embedding_stream1_stage3_before_PB, embedding_stream2_stage3_before_PB))  # [16, 1]
        stage4_delta_k = self.stage4_delta_k(
            torch.mul(embedding_stream1_stage4_before_PB, embedding_stream2_stage4_before_PB))  # [16, 1]

        embedding_stage1_after_PB = torch.mul(
            self.funsion_block_stream1_stage_1_prediction_block(embedding_stream1_stage1_before_PB),
            self.funsion_block_stream2_stage_1_prediction_block(embedding_stream2_stage1_before_PB))  # [16, 3]
        embedding_stage2_after_PB = torch.mul(
            self.funsion_block_stream1_stage_2_prediction_block(embedding_stream1_stage2_before_PB),
            self.funsion_block_stream2_stage_2_prediction_block(embedding_stream2_stage2_before_PB))  # [16, 3]
        embedding_stage3_after_PB = torch.mul(
            self.funsion_block_stream1_stage_3_prediction_block(embedding_stream1_stage3_before_PB),
            self.funsion_block_stream2_stage_3_prediction_block(embedding_stream2_stage3_before_PB))  # [16, 3]
        embedding_stage4_after_PB = torch.mul(
            self.funsion_block_stream1_stage_4_prediction_block(embedding_stream1_stage4_before_PB),
            self.funsion_block_stream2_stage_4_prediction_block(embedding_stream2_stage4_before_PB))  # [16, 3]

        embedding_stage1_after_PB = self.stage1_FC_after_PB(embedding_stage1_after_PB)  # [16, 6]
        embedding_stage2_after_PB = self.stage2_FC_after_PB(embedding_stage2_after_PB)  # [16, 6]
        embedding_stage3_after_PB = self.stage3_FC_after_PB(embedding_stage3_after_PB)  # [16, 6]
        embedding_stage4_after_PB = self.stage3_FC_after_PB(embedding_stage4_after_PB)  # [16, 6]

        prob_stage_1 = self.stage1_prob(embedding_stage1_after_PB)  # [16, 3]
        index_offset_stage1 = self.stage1_index_offsets(embedding_stage1_after_PB)

        prob_stage_2 = self.stage2_prob(embedding_stage2_after_PB)
        index_offset_stage2 = self.stage2_index_offsets(embedding_stage2_after_PB)

        prob_stage_3 = self.stage3_prob(embedding_stage3_after_PB)
        index_offset_stage3 = self.stage3_index_offsets(embedding_stage3_after_PB)
        prob_stage_4 = self.stage3_prob(embedding_stage4_after_PB)
        index_offset_stage4 = self.stage4_index_offsets(embedding_stage4_after_PB)

        stage1_regress = prob_stage_1[:, 0] * 0  # [16]
        stage2_regress = prob_stage_2[:, 0] * 0
        stage3_regress = prob_stage_3[:, 0] * 0
        stage4_regress = prob_stage_4[:, 0] * 0
        # k=1
        for index in range(self.stage_num[0]):  # stage1_regress=∑pi·(i+η)
            stage1_regress = stage1_regress + (
                    index + self.lambda_index * index_offset_stage1[:, index]) * prob_stage_1[:, index]
        stage1_regress = torch.unsqueeze(stage1_regress, 1)
        stage1_regress = stage1_regress / (self.stage_num[0] * (1 + self.lambda_delta * stage1_delta_k))
        # stage1_regress=∑pi·(i+η) / (∑sk·(1+△k))
        # k=2, stage1_regress=∑pi·(i+η) / (∑sk·(1+△k))
        for index in range(self.stage_num[1]):
            stage2_regress = stage2_regress + (
                    index + self.lambda_index * index_offset_stage2[:, index]) * prob_stage_2[:, index]
        stage2_regress = torch.unsqueeze(stage2_regress, 1)
        stage2_regress = stage2_regress / (self.stage_num[0] * (1 + self.lambda_delta * stage1_delta_k) *
                                           (self.stage_num[1] * (1 + self.lambda_delta * stage2_delta_k)))
        # k=3
        for index in range(self.stage_num[2]):
            stage3_regress = stage3_regress + (
                    index + self.lambda_index * index_offset_stage3[:, index]) * prob_stage_3[:, index]
        stage3_regress = torch.unsqueeze(stage3_regress, 1)
        stage3_regress = stage3_regress / (self.stage_num[0] * (1 + self.lambda_delta * stage1_delta_k) *
                                           (self.stage_num[1] * (1 + self.lambda_delta * stage2_delta_k)) *
                                           (self.stage_num[2] * (1 + self.lambda_delta * stage3_delta_k))
                                           )

        for index in range(self.stage_num[3]):
            stage4_regress = stage4_regress + (
                    index + self.lambda_index * index_offset_stage4[:, index]) * prob_stage_4[:, index]
        stage4_regress = torch.unsqueeze(stage4_regress, 1)
        stage4_regress = stage4_regress / (self.stage_num[0] * (1 + self.lambda_delta * stage1_delta_k) *
                                           (self.stage_num[1] * (1 + self.lambda_delta * stage2_delta_k)) *
                                           (self.stage_num[2] * (1 + self.lambda_delta * stage3_delta_k)) *
                                           (self.stage_num[3] * (1 + self.lambda_delta * stage4_delta_k))
                                           )
        regress_class = (
                                    stage1_regress + stage2_regress + stage3_regress + stage4_regress) * self.class_range  # y=∑ yk * V
        regress_class = torch.squeeze(regress_class, 1)
        return regress_class




# MobileNetV3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init



class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV3_Large(nn.Module):
    def __init__(self, num_classes=1):
        super(MobileNetV3_Large, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),
            Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1),
            Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule(40), 2),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(3, 40, 240, 80, hswish(), None, 2),
            Block(3, 80, 200, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 480, 112, hswish(), SeModule(112), 1),
            Block(3, 112, 672, 112, hswish(), SeModule(112), 1),
            Block(5, 112, 672, 160, hswish(), SeModule(160), 1),
            Block(5, 160, 672, 160, hswish(), SeModule(160), 2),
            Block(5, 160, 960, 160, hswish(), SeModule(160), 1),
        )


        self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = hswish()
        self.linear3 = nn.Linear(960, 1280)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = hswish()
        self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.hs3(self.bn3(self.linear3(out)))
        out = self.linear4(out)
        out = out.flatten(0)
        return out



class MobileNetV3_Small(nn.Module):
    def __init__(self, num_classes=1):
        super(MobileNetV3_Small, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule(16), 2),
            Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1),
            Block(5, 24, 96, 40, hswish(), SeModule(40), 2),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 120, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 144, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 288, 96, hswish(), SeModule(96), 2),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
        )


        self.conv2 = nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(576)
        self.hs2 = hswish()
        self.linear3 = nn.Linear(576, 1280)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = hswish()
        self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out,4)
        out = out.view(out.size(0), -1)
        out = self.hs3(self.bn3(self.linear3(out)))
        out = self.linear4(out)
        out = out.flatten(0)
        return out



# def test():
#     net = MobileNetV3_Small()
#     x = torch.randn(2,3,224,224)
#     y = net(x)
#     print(y.size())
#
# test()



