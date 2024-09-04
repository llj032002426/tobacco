import torch
from torch import nn
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
#AlexNet
class AlexNet(nn.Module):
    def __init__(self, num_classes=1):
        super(AlexNet, self).__init__()
        # 卷积层
        self.conv = nn.Sequential(
            # 由于LRN层已经证明无用，所以这里不写LRN
            # 第一层，输入通道数3，输出通道数96，使用11x11大小的卷积核
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),  # input:[3, 224, 224] output:[96, 55, 55]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output:[96, 27, 27]

            # 第二层，开始减小卷积核大小且增大输出通道数，从而提取更多特征
            nn.Conv2d(96, 256, 5, 1, 2),  # output: [256, 27, 27]
            nn.ReLU(),
            nn.MaxPool2d(3, 2),  # output: [256, 13, 13]
            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            # 前两个卷积层后不使用池化层来减小输入的高和宽
            nn.Conv2d(256, 384, 3, 1, 1),  # output: [384, 13, 13]
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),  # output: [384, 13, 13]
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),  # output: [256, 13, 13]
            nn.ReLU(),
            nn.MaxPool2d(3, 2)  # output: [256, 6, 6]
        )
        # 全连接层
        self.fc = nn.Sequential(
            # 第一个全连接层，输入维度是256*，输出维度是4096
            nn.Linear(256 * 8, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            # 输出层
            nn.Linear(4096, num_classes),
        )
    # 前向传播
    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output.flatten(0)


"""GoogLeNet"""
import torch.nn as nn
class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1, aux_logits=False, init_weights=False):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)
        # if self.aux_logits:
        #     self.aux1 = InceptionAux(512, num_classes)
        #     self.aux2 = InceptionAux(528, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)  # Output 1 dimension for regression
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x.flatten(0)  # Return the regression output
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


'''MobileNetV3'''
import torch.nn as nn
def Hswish(x,inplace=True):
    return x * F.relu6(x + 3., inplace=inplace) / 6.
def Hsigmoid(x,inplace=True):
    return F.relu6(x + 3., inplace=inplace) / 6.
# Squeeze-And-Excite模块
class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y=self.avg_pool(x).view(b, c)
        y=self.se(y)
        y = Hsigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
class Bottleneck(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,exp_channels,stride,se='True',nl='HS'):
        super(Bottleneck, self).__init__()
        padding = (kernel_size - 1) // 2
        if nl == 'RE':
            self.nlin_layer = F.relu6
        elif nl == 'HS':
            self.nlin_layer = Hswish
        self.stride=stride
        if se:
            self.se=SEModule(exp_channels)
        else:
            self.se=None
        self.conv1=nn.Conv2d(in_channels,exp_channels,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn1 = nn.BatchNorm2d(exp_channels)
        self.conv2=nn.Conv2d(exp_channels,exp_channels,kernel_size=kernel_size,stride=stride,
                             padding=padding,groups=exp_channels,bias=False)
        self.bn2=nn.BatchNorm2d(exp_channels)
        self.conv3=nn.Conv2d(exp_channels,out_channels,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn3=nn.BatchNorm2d(out_channels)
        # 先初始化一个空序列，之后改造其成为残差链接
        self.shortcut = nn.Sequential()
        # 只有步长为1且输入输出通道不相同时才采用跳跃连接(想一下跳跃链接的过程，输入输出通道相同这个跳跃连接就没意义了)
        if stride == 1 and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                # 下面的操作卷积不改变尺寸和通道数
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self,x):
        out=self.nlin_layer(self.bn1(self.conv1(x)))
        if self.se is not None:
            out=self.bn2(self.conv2(out))
            out=self.nlin_layer(self.se(out))
        else:
            out = self.nlin_layer(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out
class MobileNetV3_large(nn.Module):
    # (out_channels,kernel_size,exp_channels,stride,se,nl)
    cfg=[
        (16,3,16,1,False,'RE'),
        (24,3,64,2,False,'RE'),
        (24,3,72,1,False,'RE'),
        (40,5,72,2,True,'RE'),
        (40,5,120,1,True,'RE'),
        (40,5,120,1,True,'RE'),
        (80,3,240,2,False,'HS'),
        (80,3,200,1,False,'HS'),
        (80,3,184,1,False,'HS'),
        (80,3,184,1,False,'HS'),
        (112,3,480,1,True,'HS'),
        (112,3,672,1,True,'HS'),
        (160,5,672,2,True,'HS'),
        (160,5,960,1,True,'HS'),
        (160,5,960,1,True,'HS')
    ]
    def __init__(self,num_classes=1):
        super(MobileNetV3_large,self).__init__()
        self.conv1=nn.Conv2d(3,16,3,2,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(16)
        # 根据cfg数组自动生成所有的Bottleneck层
        self.layers = self._make_layers(in_channels=16)
        self.conv2=nn.Conv2d(160,960,1,stride=1,bias=False)
        self.bn2=nn.BatchNorm2d(960)
        # 卷积后不跟BN，就应该把bias设置为True
        self.conv3=nn.Conv2d(960,1280,1,1,padding=0,bias=True)
        self.conv4=nn.Conv2d(1280,num_classes,1,stride=1,padding=0,bias=True)
    def _make_layers(self,in_channels):
        layers=[]
        for out_channels,kernel_size,exp_channels,stride,se,nl in self.cfg:
            layers.append(
                Bottleneck(in_channels,out_channels,kernel_size,exp_channels,stride,se,nl)
            )
            in_channels=out_channels
        return nn.Sequential(*layers)
    def forward(self,x):
        out=Hswish(self.bn1(self.conv1(x)))
        out=self.layers(out)
        out=Hswish(self.bn2(self.conv2(out)))
        out=F.avg_pool2d(out,4)
        out=Hswish(self.conv3(out))
        out=self.conv4(out)
        # 因为原论文中最后一层是卷积层来实现全连接的效果，维度是四维的，后两维是1，在计算损失函数的时候要求二维，因此在这里需要做一个resize
        a,b=out.size(0),out.size(1)
        out=out.view(a,b)
        return out.flatten(0)
class MobileNetV3_small(nn.Module):
    # (out_channels,kernel_size,exp_channels,stride,se,nl)
    cfg = [
        (16,3,16,2,True,'RE'),
        (24,3,72,2,False,'RE'),
        (24,3,88,1,False,'RE'),
        (40,5,96,2,True,'HS'),
        (40,5,240,1,True,'HS'),
        (40,5,240,1,True,'HS'),
        (48,5,120,1,True,'HS'),
        (48,5,144,1,True,'HS'),
        (96,5,288,2,True,'HS'),
        (96,5,576,1,True,'HS'),
        (96,5,576,1,True,'HS')
    ]
    def __init__(self,num_classes=1):
        super(MobileNetV3_small,self).__init__()
        self.conv1=nn.Conv2d(3,16,3,2,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(16)
        # 根据cfg数组自动生成所有的Bottleneck层
        self.layers = self._make_layers(in_channels=16)
        self.conv2=nn.Conv2d(96,576,1,stride=1,bias=False)
        self.bn2=nn.BatchNorm2d(576)
        # 卷积后不跟BN，就应该把bias设置为True
        self.conv3=nn.Conv2d(576,1280,1,1,padding=0,bias=True)
        self.conv4=nn.Conv2d(1280,num_classes,1,stride=1,padding=0,bias=True)
    def _make_layers(self,in_channels):
        layers=[]
        for out_channels,kernel_size,exp_channels,stride,se,nl in self.cfg:
            layers.append(
                Bottleneck(in_channels,out_channels,kernel_size,exp_channels,stride,se,nl)
            )
            in_channels=out_channels
        return nn.Sequential(*layers)
    def forward(self,x):
        out=Hswish(self.bn1(self.conv1(x)))
        out=self.layers(out)
        out=self.bn2(self.conv2(out))
        se=SEModule(out.size(1))
        out=Hswish(se(out))
        out = F.avg_pool2d(out, 4)
        out = Hswish(self.conv3(out))
        out = self.conv4(out)
        # 因为原论文中最后一层是卷积层来实现全连接的效果，维度是四维的，后两维是1，在计算损失函数的时候要求二维，因此在这里需要做一个resize
        a, b = out.size(0), out.size(1)
        out = out.view(a, b)
        return out.flatten(0)




'''DenseNet'''
# This implementation is based on the DenseNet-BC implementation in torchvision
# https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
import torch.nn as nn
import torch.utils.checkpoint as cp
from collections import OrderedDict
def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output
    return bn_function
class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                        kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        self.efficient = efficient
    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features
class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))
class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)
    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)
class DenseNet(nn.Module):
    def __init__(self, growth_rate=12, block_config=(16, 16, 16), compression=0.5,
                 num_init_features=24, bn_size=4, drop_rate=0,
                 num_classes=1, small_inputs=True, efficient=False):

        super(DenseNet, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
        # First convolution
        if small_inputs:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ]))
            self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
            self.features.add_module('relu0', nn.ReLU(inplace=True))
            self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                                           ceil_mode=False))
        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)
        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))
        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)
        # Initialization
        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                n = param.size(0) * param.size(2) * param.size(3)
                param.data.normal_().mul_(math.sqrt(2. / n))
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)
            elif 'classifier' in name and 'bias' in name:
                param.data.fill_(0)
    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out.flatten(0)


'''GhostNet'''
import torch.nn as nn
import torch.nn.functional as F
import math
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.
class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)
    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x
class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x
class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]
class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""
    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0.):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride
        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)
        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                     padding=(dw_kernel_size - 1) // 2,
                                     groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)
        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None
        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)
        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )
    def forward(self, x):
        residual = x
        # 1st ghost bottleneck
        x = self.ghost1(x)
        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)
        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)
        # 2nd ghost bottleneck
        x = self.ghost2(x)
        x += self.shortcut(residual)
        return x
class GhostNet(nn.Module):
    def __init__(self, cfgs, num_classes=1, width=1.0, dropout=0.2):
        super(GhostNet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.dropout = dropout
        # building first layer
        output_channel = _make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel
        # building inverted residual blocks
        stages = []
        block = GhostBottleneck
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                layers.append(block(input_channel, hidden_channel, output_channel, k, s,
                                    se_ratio=se_ratio))
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))
        output_channel = _make_divisible(exp_size * width, 4)
        stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
        input_channel = output_channel
        self.blocks = nn.Sequential(*stages)
        # building last several layers
        output_channel = 1280
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_head = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True)
        self.act2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(output_channel, num_classes)
    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = x.view(x.size(0), -1)
        if self.dropout > 0.:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x.flatten(0)
def ghostnet(**kwargs):
    cfgs = [
        # k, t, c, SE, s
        # stage1
        [[3, 16, 16, 0, 1]],
        # stage2
        [[3, 48, 24, 0, 2]],
        [[3, 72, 24, 0, 1]],
        # stage3
        [[5, 72, 40, 0.25, 2]],
        [[5, 120, 40, 0.25, 1]],
        # stage4
        [[3, 240, 80, 0, 2]],
        [[3, 200, 80, 0, 1],
         [3, 184, 80, 0, 1],
         [3, 184, 80, 0, 1],
         [3, 480, 112, 0.25, 1],
         [3, 672, 112, 0.25, 1]
         ],
        # stage5
        [[5, 672, 160, 0.25, 2]],
        [[5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1],
         [5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1]
         ]
    ]
    return GhostNet(cfgs, **kwargs)


'''Csp_ResNet'''
import torch
from torchvision.models.resnet import conv1x1,conv3x3,BasicBlock,Bottleneck,ResNet
def _downsample(inplanes,outplanes,stride):
    return torch.nn.Sequential(
        conv1x1(inplanes, outplanes, stride),
        torch.nn.BatchNorm2d(outplanes),
    )
class Csp_ResNet(ResNet):
    def __init__(self, block, layers, num_classes=1, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None,channldown = True):
        self.expansion = block.expansion
        self.layerplanes = [64]
        self.CspChanneldown = 2 if channldown else 1
        super(Csp_ResNet,self).__init__(block, layers, num_classes, zero_init_residual,
                 groups, width_per_group, replace_stride_with_dilation, norm_layer)
        self.downlayer1 = torch.nn.Identity()
        self.downlayer2 = _downsample(self.layerplanes[1]//2,self.layerplanes[1]//2,2)
        self.downlayer3 = _downsample(self.layerplanes[2]//2,self.layerplanes[2]//2,2)
        self.downlayer4 = _downsample(self.layerplanes[3]//2,self.layerplanes[3]//2,2)
        self.fc = torch.nn.Linear(self.layerplanes[4],num_classes)
    def _make_layer(self,block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        inplanes = self.inplanes - self.inplanes//2
        outplanes = (planes //self.CspChanneldown) * block.expansion
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or inplanes != outplanes:
            downsample = torch.nn.Sequential(
                conv1x1(inplanes, outplanes, stride),
                norm_layer(outplanes),
            )
        layers = []
        layers.append(block(inplanes, planes//self.CspChanneldown, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        inplanes = outplanes
        for _ in range(1,blocks):
            layers.append(block(inplanes, planes//self.CspChanneldown, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        self.inplanes = self.inplanes//2 +  outplanes
        self.layerplanes.append(self.inplanes)
        return torch.nn.Sequential(*layers)
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = torch.cat([self.downlayer1(x[:,:self.layerplanes[0]//2,...]),self.layer1(x[:,self.layerplanes[0]//2:,...])],1)
        x = torch.cat([self.downlayer2(x[:,:self.layerplanes[1]//2,...]),self.layer2(x[:,self.layerplanes[1]//2:,...])],1)
        x = torch.cat([self.downlayer3(x[:,:self.layerplanes[2]//2,...]),self.layer3(x[:,self.layerplanes[2]//2:,...])],1)
        x = torch.cat([self.downlayer4(x[:,:self.layerplanes[3]//2,...]),self.layer4(x[:,self.layerplanes[3]//2:,...])],1)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x.flatten(0)
def _csp_resnet(block, layers, **kwargs):
    model = Csp_ResNet(block,layers,**kwargs)
    return model
def csp_resnet50(**kwargs):
    return _csp_resnet(Bottleneck,[3, 4, 6, 3],**kwargs)



'''efficientnetv2'''
import torch
import torch.nn as nn
import math
__all__ = ['effnetv2_s', 'effnetv2_m', 'effnetv2_l', 'effnetv2_xl']
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
# SiLU (Swish) activation function
if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
else:
    # For compatibility with old PyTorch versions
    class SiLU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)
class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, _make_divisible(inp // reduction, 8)),
            SiLU(),
            nn.Linear(_make_divisible(inp // reduction, 8), oup),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        SiLU()
    )
def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        SiLU()
    )
class MBConv(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_se):
        super(MBConv, self).__init__()
        assert stride in [1, 2]
        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if use_se:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                SELayer(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # fused
                nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)
class EffNetV2(nn.Module):
    def __init__(self, cfgs, num_classes=1, width_mult=1.):
        super(EffNetV2, self).__init__()
        self.cfgs = cfgs
        # building first layer
        input_channel = _make_divisible(24 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = MBConv
        for t, c, n, s, use_se in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1792 * width_mult, 8) if width_mult > 1.0 else 1792
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)
        self._initialize_weights()
    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x.flatten(0)
    def _initialize_weights(self):
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
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()
def effnetv2_s(**kwargs):
    cfgs = [
        # t, c, n, s, SE
        [1, 24, 2, 1, 0],
        [4, 48, 4, 2, 0],
        [4, 64, 4, 2, 0],
        [4, 128, 6, 2, 1],
        [6, 160, 9, 1, 1],
        [6, 256, 15, 2, 1],
    ]
    return EffNetV2(cfgs, **kwargs)
def effnetv2_m(**kwargs):
    cfgs = [
        # t, c, n, s, SE
        [1, 24, 3, 1, 0],
        [4, 48, 5, 2, 0],
        [4, 80, 5, 2, 0],
        [4, 160, 7, 2, 1],
        [6, 176, 14, 1, 1],
        [6, 304, 18, 2, 1],
        [6, 512, 5, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)
def effnetv2_l(**kwargs):
    cfgs = [
        # t, c, n, s, SE
        [1, 32, 4, 1, 0],
        [4, 64, 7, 2, 0],
        [4, 96, 7, 2, 0],
        [4, 192, 10, 2, 1],
        [6, 224, 19, 1, 1],
        [6, 384, 25, 2, 1],
        [6, 640, 7, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)
def effnetv2_xl(**kwargs):
    cfgs = [
        # t, c, n, s, SE
        [1, 32, 4, 1, 0],
        [4, 64, 8, 2, 0],
        [4, 96, 8, 2, 0],
        [4, 192, 16, 2, 1],
        [6, 256, 24, 1, 1],
        [6, 512, 32, 2, 1],
        [6, 640, 8, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


"""Darknet-53"""
import torch
from torch import nn
def conv_batch(in_num, out_num, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_num),
        nn.LeakyReLU())
# Residual block
class DarkResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(DarkResidualBlock, self).__init__()
        reduced_channels = int(in_channels/2)
        self.layer1 = conv_batch(in_channels, reduced_channels, kernel_size=1, padding=0)
        self.layer2 = conv_batch(reduced_channels, in_channels)
    def forward(self, x):
        residual = x
        out = self.layer1(x)
        out = self.layer2(out)
        out += residual
        return out
class Darknet53(nn.Module):
    def __init__(self, input_channel=3, n_classes=1):
        super(Darknet53, self).__init__()
        self.conv1 = conv_batch(input_channel, 32)
        self.conv2 = conv_batch(32, 64, stride=2)
        self.residual_block1 = self.make_layer(block = DarkResidualBlock, in_channels=64, num_blocks=1)
        self.conv3 = conv_batch(64, 128, stride=2)
        self.residual_block2 = self.make_layer(block = DarkResidualBlock, in_channels=128, num_blocks=2)
        self.conv4 = conv_batch(128, 256, stride=2)
        self.residual_block3 = self.make_layer(block = DarkResidualBlock, in_channels=256, num_blocks=8)
        self.conv5 = conv_batch(256, 512, stride=2)
        self.residual_block4 = self.make_layer(block = DarkResidualBlock, in_channels=512, num_blocks=8)
        self.conv6 = conv_batch(512, 1024, stride=2)
        self.residual_block5 = self.make_layer(block = DarkResidualBlock, in_channels=1024, num_blocks=4)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, n_classes)
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.residual_block1(out)
        out = self.conv3(out)
        out = self.residual_block2(out)
        out = self.conv4(out)
        out = self.residual_block3(out)
        out = self.conv5(out)
        out = self.residual_block4(out)
        out = self.conv6(out)
        out = self.residual_block5(out)
        out = self.global_avg_pool(out)
        out = out.view(-1, 1024)
        out = self.fc(out)
        return out.flatten(0)
    def make_layer(self, in_channels, num_blocks, block):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels))
        return nn.Sequential(*layers)


"""InceptionV3"""
import torch
import torch.nn as nn
class BasicConvBlock(nn.Module):
    def __init__(self, input_channel, output_channel, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, bias=False, **kwargs),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )
    def forward(self, X):
        return self.conv(X)
#Inception Block from InceptionV1
class InceptionA(nn.Module):
    def __init__(self, input_channel, pool_features):
        super().__init__()
        self.branch1x1 = BasicConvBlock(input_channel, 64, kernel_size=1)
        self.branch5x5 = nn.Sequential(
            BasicConvBlock(input_channel, 48, kernel_size=1),
            BasicConvBlock(48, 64, kernel_size=5, padding=2)
        )
        self.branch3x3 = nn.Sequential(
            BasicConvBlock(input_channel, 64, kernel_size=1),
            BasicConvBlock(64, 96, kernel_size=3, padding=1),
            BasicConvBlock(96, 96, kernel_size=3, padding=1)
        )
        self.branchpool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConvBlock(input_channel, pool_features, kernel_size=3, padding=1)
        )
    def forward(self, X):
         #x -> 1x1(same)
        branch1x1 = self.branch1x1(X)
        #x -> 1x1 -> 5x5(same)
        branch5x5 = self.branch5x5(X)
        #branch5x5 = self.branch5x5_2(branch5x5)
        #x -> 1x1 -> 3x3 -> 3x3(same)
        branch3x3 = self.branch3x3(X)
        #x -> pool -> 1x1(same)
        branchpool = self.branchpool(X)
        outputs = [branch1x1, branch5x5, branch3x3, branchpool]
        return torch.cat(outputs, 1)
#Factorization
class InceptionB(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.branch3x3 = BasicConvBlock(input_channels, 384, kernel_size=3, stride=2)
        self.branch3x3stack = nn.Sequential(
            BasicConvBlock(input_channels, 64, kernel_size=1),
            BasicConvBlock(64, 96, kernel_size=3, padding=1),
            BasicConvBlock(96, 96, kernel_size=3, stride=2)
        )
        self.branchpool = nn.MaxPool2d(kernel_size=3, stride=2)
    def forward(self, X):
        #x - > 3x3(downsample)
        branch3x3 = self.branch3x3(X)
        #x -> 3x3 -> 3x3(downsample)
        branch3x3stack = self.branch3x3stack(X)
        #x -> avgpool(downsample)
        branchpool = self.branchpool(X)
        outputs = [branch3x3, branch3x3stack, branchpool]
        return torch.cat(outputs, 1)
#Factorizing Convolutions with Large Filter Size
class InceptionC(nn.Module):
    def __init__(self, input_channels, channels_7x7):
        super().__init__()
        self.branch1x1 = BasicConvBlock(input_channels, 192, kernel_size=1)
        c7 = channels_7x7
        #In theory, we could go even further and argue that one can replace any n × n
        #convolution by a 1 × n convolution followed by a n × 1 convolution and the
        #computational cost saving increases dramatically as n grows (see figure 6).
        self.branch7x7 = nn.Sequential(
            BasicConvBlock(input_channels, c7, kernel_size=1),
            BasicConvBlock(c7, c7, kernel_size=(7, 1), padding=(3, 0)),
            BasicConvBlock(c7, 192, kernel_size=(1, 7), padding=(0, 3))
        )
        self.branch7x7stack = nn.Sequential(
            BasicConvBlock(input_channels, c7, kernel_size=1),
            BasicConvBlock(c7, c7, kernel_size=(7, 1), padding=(3, 0)),
            BasicConvBlock(c7, c7, kernel_size=(1, 7), padding=(0, 3)),
            BasicConvBlock(c7, c7, kernel_size=(7, 1), padding=(3, 0)),
            BasicConvBlock(c7, 192, kernel_size=(1, 7), padding=(0, 3))
        )
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConvBlock(input_channels, 192, kernel_size=1),
        )
    def forward(self, x):
        #x -> 1x1(same)
        branch1x1 = self.branch1x1(x)
        #x -> 1layer 1*7 and 7*1 (same)
        branch7x7 = self.branch7x7(x)
        #x-> 2layer 1*7 and 7*1(same)
        branch7x7stack = self.branch7x7stack(x)
        #x-> avgpool (same)
        branchpool = self.branch_pool(x)
        outputs = [branch1x1, branch7x7, branch7x7stack, branchpool]
        return torch.cat(outputs, 1)
class InceptionD(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.branch3x3 = nn.Sequential(
            BasicConvBlock(input_channels, 192, kernel_size=1),
            BasicConvBlock(192, 320, kernel_size=3, stride=2)
        )
        self.branch7x7 = nn.Sequential(
            BasicConvBlock(input_channels, 192, kernel_size=1),
            BasicConvBlock(192, 192, kernel_size=(1, 7), padding=(0, 3)),
            BasicConvBlock(192, 192, kernel_size=(7, 1), padding=(3, 0)),
            BasicConvBlock(192, 192, kernel_size=3, stride=2)
        )
        self.branchpool = nn.AvgPool2d(kernel_size=3, stride=2)
    def forward(self, x):
        #x -> 1x1 -> 3x3(downsample)
        branch3x3 = self.branch3x3(x)
        #x -> 1x1 -> 1x7 -> 7x1 -> 3x3 (downsample)
        branch7x7 = self.branch7x7(x)
        #x -> avgpool (downsample)
        branchpool = self.branchpool(x)
        outputs = [branch3x3, branch7x7, branchpool]
        return torch.cat(outputs, 1)
#same
class InceptionE(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.branch1x1 = BasicConvBlock(input_channels, 320, kernel_size=1)
        self.branch3x3_1 = BasicConvBlock(input_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConvBlock(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConvBlock(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch3x3stack_1 = BasicConvBlock(input_channels, 448, kernel_size=1)
        self.branch3x3stack_2 = BasicConvBlock(448, 384, kernel_size=3, padding=1)
        self.branch3x3stack_3a = BasicConvBlock(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3stack_3b = BasicConvBlock(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConvBlock(input_channels, 192, kernel_size=1)
        )
    def forward(self, x):
        #x -> 1x1 (same)
        branch1x1 = self.branch1x1(x)
        # x -> 1x1 -> 3x1
        # x -> 1x1 -> 1x3
        # concatenate(3x1, 1x3)
        #"""7. Inception modules with expanded the filter bank outputs.
        #This architecture is used on the coarsest (8 × 8) grids to promote
        #high dimensional representations, as suggested by principle
        #2 of Section 2."""
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3)
        ]
        branch3x3 = torch.cat(branch3x3, 1)
        # x -> 1x1 -> 3x3 -> 1x3
        # x -> 1x1 -> 3x3 -> 3x1
        #concatenate(1x3, 3x1)
        branch3x3stack = self.branch3x3stack_1(x)
        branch3x3stack = self.branch3x3stack_2(branch3x3stack)
        branch3x3stack = [
            self.branch3x3stack_3a(branch3x3stack),
            self.branch3x3stack_3b(branch3x3stack)
        ]
        branch3x3stack = torch.cat(branch3x3stack, 1)
        branchpool = self.branch_pool(x)
        outputs = [branch1x1, branch3x3, branch3x3stack, branchpool]
        return torch.cat(outputs, 1)
class InceptionV3(nn.Module):
    def __init__(self, input_channel=3, n_classes=1):
        super().__init__()
        self.Conv2d_1a_3x3 = BasicConvBlock(input_channel, 32, kernel_size=3, padding=1)
        self.Conv2d_2a_3x3 = BasicConvBlock(32, 32, kernel_size=3, padding=1)
        self.Conv2d_2b_3x3 = BasicConvBlock(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConvBlock(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConvBlock(80, 192, kernel_size=3)
        #naive inception module
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        #downsample
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)
        #downsample
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)
        #6*6 feature size
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d()
        self.linear = nn.Linear(2048, n_classes)
    def forward(self, x):
        #32 -> 30
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        #30 -> 30
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        #30 -> 14
        #Efficient Grid Size Reduction to avoid representation
        #bottleneck
        x = self.Mixed_6a(x)
        #14 -> 14
        #"""In practice, we have found that employing this factorization does not
        #work well on early layers, but it gives very good results on medium
        #grid-sizes (On m × m feature maps, where m ranges between 12 and 20).
        #On that level, very good results can be achieved by using 1 × 7 convolutions
        #followed by 7 × 1 convolutions."""
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        #14 -> 6
        #Efficient Grid Size Reduction
        x = self.Mixed_7a(x)
        #6 -> 6
        #We are using this solution only on the coarsest grid,
        #since that is the place where producing high dimensional
        #sparse representation is the most critical as the ratio of
        #local processing (by 1 × 1 convolutions) is increased compared
        #to the spatial aggregation."""
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        #6 -> 1
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x.flatten(0)


"""SqueezeNet"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
class Fire(nn.Module):
    def __init__(self, in_channel, squeeze_channel, expand_channel):
        super().__init__()
        # squeeze conv1x1
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channel, squeeze_channel, kernel_size=1, stride=1),
            nn.BatchNorm2d(squeeze_channel),
            nn.ReLU(inplace=True),
        )
        # expand conv1x1
        self.expand1x1 = nn.Sequential(
            nn.Conv2d(squeeze_channel, expand_channel, kernel_size=1, stride=1),
            nn.BatchNorm2d(expand_channel)
        )
        # expand conv3x3
        self.expand3x3 = nn.Sequential(
            nn.Conv2d(squeeze_channel, expand_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(expand_channel)
        )
        # using MSR initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
    def forward(self, x):
        squeezeOut = self.squeeze(x)
        expandOut_1x1 = self.expand1x1(squeezeOut)
        expandOut_3x3 = self.expand3x3(squeezeOut)
        output = torch.cat([expandOut_1x1, expandOut_3x3], 1)
        output = F.relu(output)
        return output
class SqueezeNet(nn.Module):
    def __init__(self, input_channel=3, n_classes=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channel, 96, kernel_size=3, stride=1, padding=1),  # 32
            nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 16
        )
        self.Fire2 = Fire(96, 16, 64)
        self.Fire3 = Fire(128, 16, 64)
        self.Fire4 = Fire(128, 32, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 8
        self.Fire5 = Fire(256, 32, 128)
        self.Fire6 = Fire(256, 48, 192)
        self.Fire7 = Fire(384, 48, 192)
        self.Fire8 = Fire(384, 64, 256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 4
        self.Fire9 = Fire(512, 64, 256)
        self.conv2 = nn.Conv2d(512, n_classes, kernel_size=1, stride=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.LogSoftmax(dim=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, x):
        x = self.conv(x)
        x = self.Fire2(x)
        x = self.Fire3(x)
        x = self.Fire4(x)
        x = self.maxpool2(x)
        x = self.Fire5(x)
        x = self.Fire6(x)
        x = self.Fire7(x)
        x = self.Fire8(x)
        x = self.maxpool3(x)
        x = self.Fire9(x)
        x = F.dropout(x, 0.5)
        x = self.conv2(x)
        x = self.avg_pool(x)
        x = self.softmax(x)
        x = x.view(x.size(0), -1)
        return x.flatten(0)



def test():
    x = torch.randn(16, 3, 120, 170)
    # alex_model = AlexNet()
    # alex_model = GoogLeNet()
    # alex_model = MobileNetV3_large()
    # alex_model = DenseNet()
    # alex_model = ghostnet()
    # alex_model = csp_resnet50()
    # alex_model = effnetv2_xl()
    # alex_model = Darknet53()
    # alex_model = InceptionV3()
    alex_model = SqueezeNet()
    pred = alex_model(x)
    print(x.shape)
    print(pred.shape)


if __name__ == '__main__':
    test()