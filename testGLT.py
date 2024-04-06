"""
This is the code for global-local transformer for brain age estimation
@email: heshengxgd@gmail.com
"""

import torch
import torch.nn as nn

import copy
import math
import cv2
from textural_feature import fast_glcm_mean,fast_glcm_std,fast_glcm_contrast,fast_glcm_dissimilarity,fast_glcm_homogeneity,fast_glcm_ASM,fast_glcm_ENE,fast_glcm_max,fast_glcm_entropy,all_glcm
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
class convBlock(nn.Module): #每个模块
    def __init__(self, inplace, outplace, kernel_size=3, padding=1):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplace, outplace, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(outplace)

    def forward(self, x):
        x = x.to(device)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x


class VGG16(nn.Module):
    def __init__(self, inplace):
        super().__init__()

        ly = [64, 128, 256, 512, 512]

        self.ly = ly

        self.maxp = nn.MaxPool2d(2)

        self.conv11 = convBlock(inplace, ly[0])
        self.conv12 = convBlock(ly[0], ly[0])

        self.conv21 = convBlock(ly[0], ly[1])
        self.conv22 = convBlock(ly[1], ly[1])

        self.conv31 = convBlock(ly[1], ly[2])
        self.conv32 = convBlock(ly[2], ly[2])
        self.conv33 = convBlock(ly[2], ly[2])

        self.conv41 = convBlock(ly[2], ly[3])
        self.conv42 = convBlock(ly[3], ly[3])
        self.conv43 = convBlock(ly[3], ly[3])

        self.conv51 = convBlock(ly[3], ly[3])
        self.conv52 = convBlock(ly[3], ly[3])
        self.conv53 = convBlock(ly[3], ly[3])

    def forward(self, x):
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.maxp(x)

        x = self.conv21(x)
        x = self.conv22(x)
        x = self.maxp(x)

        x = self.conv31(x)
        x = self.conv32(x)
        x = self.conv33(x)
        x = self.maxp(x)

        x = self.conv41(x)
        x = self.conv42(x)
        x = self.conv43(x)
        x = self.maxp(x)

        x = self.conv51(x)
        x = self.conv52(x)
        x = self.conv53(x)

        return x
class VGG8(nn.Module):
    def __init__(self, inplace):
        super().__init__()

        ly = [64, 128, 256, 512]#每个模块的通道数

        self.ly = ly

        self.maxp = nn.MaxPool2d(2)

        self.conv11 = convBlock(inplace, ly[0])#模块
        self.conv12 = convBlock(ly[0], ly[0])

        self.conv21 = convBlock(ly[0], ly[1])
        self.conv22 = convBlock(ly[1], ly[1])

        self.conv31 = convBlock(ly[1], ly[2])
        self.conv32 = convBlock(ly[2], ly[2])

        self.conv41 = convBlock(ly[2], ly[3])
        self.conv42 = convBlock(ly[3], ly[3])

    def forward(self, x):
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.maxp(x)

        x = self.conv21(x)
        x = self.conv22(x)
        x = self.maxp(x)

        x = self.conv31(x)
        x = self.conv32(x)
        x = self.maxp(x)

        x = self.conv41(x)
        x = self.conv42(x)
        x = self.maxp(x)

        return x
class GlobalAttention(nn.Module):
    def __init__(self,
                 transformer_num_heads=8,
                 hidden_size=512,
                 transformer_dropout_rate=0.0):
        super().__init__()

        self.num_attention_heads = transformer_num_heads #注意力头的数量
        self.attention_head_size = int(hidden_size / self.num_attention_heads) #注意力头的大小，即隐藏层大小除以注意力头的数量
        self.all_head_size = self.num_attention_heads * self.attention_head_size #所有头的总大小

        self.query = nn.Linear(hidden_size, self.all_head_size) #创建一个线性层（全连接层），用于将输入特征映射到查询向量空间
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.out = nn.Linear(hidden_size, hidden_size) #创建一个线性层，用于将注意力机制的输出映射回隐藏层的大小
        self.attn_dropout = nn.Dropout(transformer_dropout_rate) #创建一个dropout层，用于在计算注意力时进行随机丢弃
        self.proj_dropout = nn.Dropout(transformer_dropout_rate) #创建另一个dropout层，用于在投影过程中进行随机丢弃

        self.softmax = nn.Softmax(dim=-1) # 创建一个softmax层，用于计算注意力分数，dim=-1表示对最后一个维度进行softmax操作

    def transpose_for_scores(self, x): # 定义了一个辅助函数，用于将线性变换后的张量重新排列成注意力分数矩阵的形状
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size) #计算新的张量形状，除了最后两个维度外，其余维度保持不变
        # x.size()[:-1]这表示对张量 x 的大小（即维度）进行切片操作，切片的范围是从第一个维度开始一直到倒数第二个维度。例如，如果 x 是一个形状为 (3, 4, 5, 6) 的张量，则 x.size()[:-1] 的结果将是 (3, 4, 5)
        #这是在切片操作的基础上，将其余维度保持不变，并在最后添加两个新的维度，分别是 self.num_attention_heads 和 self.attention_head_size。这样就将原始张量的形状扩展为一个新的形状元组。
        x = x.view(*new_x_shape) #将张量按新的形状进行重塑
        #这是一个张量的重塑操作，其中 new_x_shape 是一个包含新形状的元组。view 方法将张量重塑成指定形状的张量。在这里，new_x_shape 是一个包含新形状的元组，通过 * 运算符展开成位置参数，传递给 view 方法，从而将张量 x 重塑成新的形状
        return x.permute(0, 2, 1, 3) #将张量的维度进行置换，以符合注意力分数的形状
        #这是一个维度置换操作，通过指定新的维度顺序来重新排列张量的维度。在这个例子中，原始张量 x 的维度顺序是 (batch_size, seq_length, num_attention_heads, attention_head_size)，
        # 通过 permute 方法，我们将其转换为 (batch_size, num_attention_heads, seq_length, attention_head_size)

    def forward(self, locx, glox):#定义了模型的前向传播函数，接受两个输入参数：locx（局部特征）和glox（全局特征）
        locx_query_mix = self.query(locx) #通过query线性层将局部特征映射到查询向量空间
        glox_key_mix = self.key(glox) #通过key线性层将全局特征映射到键向量空间
        glox_value_mix = self.value(glox) #通过value线性层将全局特征映射到值向量空间

        query_layer = self.transpose_for_scores(locx_query_mix) #将查询向量重排成注意力分数矩阵的形状
        key_layer = self.transpose_for_scores(glox_key_mix)
        value_layer = self.transpose_for_scores(glox_value_mix)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) #计算注意力分数，即查询向量与键向量的点积
        attention_scores = attention_scores / math.sqrt(self.attention_head_size) #对注意力分数进行缩放操作，以减小较大的分数
        attention_probs = self.softmax(attention_scores) #对注意力分数进行softmax操作，将其转换为概率分布

        attention_probs = self.attn_dropout(attention_probs) #在注意力概率上进行dropout操作，以增加模型的鲁棒性
        context_layer = torch.matmul(attention_probs, value_layer) #将注意力概率与值向量相乘，得到上下文向量

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() #对上下文向量的维度进行置换，以便后续的操作
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,) # 计算新的上下文向量的形状
        context_layer = context_layer.view(*new_context_layer_shape) #将上下文向量重塑成新的形状

        attention_output = self.out(context_layer) #将上下文向量映射回原始隐藏层大小的空间
        attention_output = self.proj_dropout(attention_output) #在输出上进行dropout操作，以增加模型的鲁棒性

        return attention_output
class Feedforward(nn.Module):
    def __init__(self, inplace, outplace):
        super().__init__()

        self.conv1 = convBlock(inplace, outplace, kernel_size=1, padding=0)
        self.conv2 = convBlock(outplace, outplace, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
class HistogramBlock(nn.Module):
    def __init__(self):
        super(HistogramBlock, self).__init__()

    def forward(self, x):
        # batch_size, channels, height, width = x.size()
        # hist_features = []
        # for i in range(batch_size):
        #     hist_list = []
        #     for c in range(channels):
        #         channel_hist = cv2.calcHist([x[i][c].cpu().numpy()], [0], None, [256], [0, 256])
        #         channel_hist = torch.tensor(channel_hist)
        #         hist_list.append(channel_hist)
        #     hist_tensor = torch.stack(hist_list, dim=0)
        #     hist_features.append(hist_tensor)
        # hist_features = torch.stack(hist_features, dim=0)

        batch_size, channels, height, width = x.size()
        # hist_features = torch.zeros(batch_size, 3, 16, 16)
        hist_features = torch.zeros(batch_size, 3, height, width)
        for i in range(batch_size):
            # 计算图像的直方图
            # src = x[i].cpu().numpy()
            # src = np.transpose(src, (1, 2, 0))
            # src = (src * 255).astype(np.uint8)
            # print(src)
            # hist_0 = cv2.calcHist([src], [0], None, [256], [0, 1])
            # print(hist_0)
            # hist_0 = torch.Tensor(hist_0)
            hist_0 = torch.histc(x[i, 0, :, :], bins=256, min=0, max=1).unsqueeze(1)
            # print(hist_0)
            hist_1 = torch.histc(x[i, 1, :, :], bins=256, min=0, max=1).unsqueeze(1)
            hist_2 = torch.histc(x[i, 2, :, :], bins=256, min=0, max=1).unsqueeze(1)
            # 将三个通道的直方图堆叠起来，形成一个形状为 (3, 256, 1) 的张量
            hist = torch.stack((hist_0, hist_1, hist_2), dim=0)
            # print(hist)

            hist = hist.unsqueeze(0)
            hist = torch.nn.functional.interpolate(hist, size=(height, width), mode='bilinear', align_corners=False)
            hist = hist.squeeze()

            # hist = torch.reshape(hist, (-1, 16, 16))
            hist_features[i] = hist
        return hist_features
class TextureBlock(nn.Module):
    def __init__(self):
        super(TextureBlock, self).__init__()
    def forward(self, x):
        # batch_size, channels, height, width = x.size()
        # text_features = []
        # for i in range(batch_size):
        #     text_list = []
        #     for c in range(channels):
        #         channel_text = fast_glcm_ASM(x[i][c].cpu().numpy())
        #         channel_text = torch.tensor(channel_text)
        #         text_list.append(channel_text)
        #     text_tensor = torch.stack(text_list, dim=0)
        #     text_features.append(text_tensor)
        # text_features = torch.stack(text_features, dim=0)

        batch_size, channels, height, width = x.size()
        # text_features = torch.zeros(batch_size, 1, 16, 16)
        text_features = torch.zeros(batch_size, 1, height, width)
        for i in range(batch_size):
            y = transforms.ToPILImage()(x[i])  # 将torch.Tensor类型图像转换为PIL图像
            # 将PIL图像转换为灰度图像
            y = y.convert("L")
            # 将灰度图像转换为NumPy数组
            y = np.array(y)
            text = fast_glcm_ASM(y)
            text = torch.from_numpy(text)
            text = text.unsqueeze(0).unsqueeze(0)
            text = torch.nn.functional.interpolate(text, size=(height, width), mode='bilinear', align_corners=False)
            text_features[i] = text
        return text_features
class feature(nn.Module):
    def __init__(self):
        super().__init__()
        # 将卷积层修改为直方图和纹理特征提取层
        self.hist1 = HistogramBlock()  # 假设您实现了HistogramBlock
        self.tex1 = TextureBlock()  # 假设您实现了TextureBlock
        self.vgg = VGG8(inplace=4)

    def forward(self, x):
        # 提取直方图和纹理特征
        hist1 = self.hist1(x) #[16, 3, height, width]
        # print(hist1)
        tex1 = self.tex1(x) #[16, 1, height, width]
        # print(tex1)
        # tex_size = tex1.size()[2:]
        # hist1 = torch.nn.functional.interpolate(hist1, size=tex_size, mode='bilinear',align_corners=False)
        x = torch.cat([hist1, tex1], dim=1)  # 在每个池化层之前连接直方图和纹理特征
        x = self.vgg(x)

        # x = self.vgg(hist1)
        return x
class GlobalLocalBrainAge(nn.Module):
    def __init__(self, inplace,
                 patch_size=64,
                 step=-1,
                 nblock=6,
                 drop_rate=0.5):
        """
        Parameter:
            @patch_size: the patch size of the local pathway
            @step: the step size of the sliding window of the local patches
            @nblock: the number of blocks for the Global-Local Transformer
            @Drop_rate: dropout rate
            @backbone: the backbone of extract the features
        """

        super().__init__()

        self.patch_size = patch_size
        self.step = step
        self.nblock = nblock

        if self.step <= 0:
            self.step = int(patch_size // 2)

        # self.global_feat = feature()  # 基准模型VGG8，把图像输入转化为深层特征
        # self.local_feat = feature()
        # hidden_size = 512

        hidden_size = 512
        self.global_feat = VGG8(inplace)  # 基准模型VGG8，把图像输入转化为深层特征
        self.local_feat = VGG8(inplace)

        # self.global_feat = VGG16(inplace)
        # self.local_feat = VGG16(inplace)

        # if backbone == 'vgg8':
        #     self.global_feat = VGG8(inplace) #基准模型VGG8，把图像输入转化为深层特征
        #     self.local_feat = VGG8(inplace)
        #     hidden_size = 512
        # elif backbone == 'vgg16':
        #     self.global_feat = VGG16(inplace)
        #     self.local_feat = VGG16(inplace)
        #     hidden_size = 512
        # else:
        #     raise ValueError('% model does not supported!' % backbone)

        self.attnlist = nn.ModuleList()
        self.fftlist = nn.ModuleList()

        for n in range(nblock):#n个全局局部转换器用于迭代地融合全局和局部特征
            atten = GlobalAttention(
                transformer_num_heads=8,
                hidden_size=hidden_size,
                transformer_dropout_rate=drop_rate)
            self.attnlist.append(atten)

            fft = Feedforward(inplace=hidden_size * 2,
                              outplace=hidden_size) #前向传播网络
            self.fftlist.append(fft)

        self.avg = nn.AdaptiveAvgPool2d(1)
        out_hidden_size = hidden_size

        self.gloout = nn.Linear(out_hidden_size, 1) #将输入特征的大小调整为1
        self.locout = nn.Linear(out_hidden_size, 1)

    def forward(self, xinput):
        _, _, H, W = xinput.size() #获取了输入张量xinput的大小，并将其分配给变量H和W，这里的"_"表示忽略的返回值
        outlist = []

        xglo = self.global_feat(xinput) #返回全局特征张量xglo
        # print(xglo)
        xgfeat = torch.flatten(self.avg(xglo), 1)#首先将xglo通过自适应平均池化层self.avg进行池化，然后使用torch.flatten函数将结果展平为一维张量
        # print(xgfeat)
        glo = self.gloout(xgfeat)
        # print(glo)

        outlist = [glo]

        B2, C2, H2, W2 = xglo.size()
        xglot = xglo.view(B2, C2, H2 * W2)
        xglot = xglot.permute(0, 2, 1)

        for y in range(0, H - self.patch_size, self.step): #遍历输入张量xinput的高度方向，并且每次迭代中变量y增加self.step
            for x in range(0, W - self.patch_size, self.step): #遍历输入张量xinput的宽度方向，并且每次迭代中变量x增加self.step
                locx = xinput[:, :, y:y + self.patch_size, x:x + self.patch_size] #从输入张量xinput中提取了一个局部区域
                xloc = self.local_feat(locx) #返回局部特征张量xloc

                for n in range(self.nblock):
                    B1, C1, H1, W1 = xloc.size()
                    xloct = xloc.view(B1, C1, H1 * W1)
                    xloct = xloct.permute(0, 2, 1)

                    tmp = self.attnlist[n](xloct, xglot) #调用了名为attnlist中第n个位置的GlobalAttention对象，传入全局特征张量xglot和局部特征张量xloct，并返回注意力权重张量attention
                    tmp = tmp.permute(0, 2, 1)
                    tmp = tmp.view(B1, C1, H1, W1)
                    tmp = torch.cat([tmp, xloc], 1)

                    tmp = self.fftlist[n](tmp)
                    xloc = xloc + tmp

                xloc = torch.flatten(self.avg(xloc), 1)

                out = self.locout(xloc)
                outlist.append(out)

        return outlist
        # return outlist[0].flatten(0)


if __name__ == '__main__':
    # x1 = torch.rand(1, 5, 130, 170)
    x1 = torch.rand(16, 3, 170, 120)
    # x1 = torch.rand(16, 3, 16, 16)

    # mod = GlobalLocalBrainAge(5,
    #                           patch_size=64,
    #                           step=32,
    #                           nblock=6,
    #                           backbone='vgg8')
    mod = GlobalLocalBrainAge(3,
                              patch_size=64,
                              step=32,
                              nblock=6,
                              backbone='vgg8')
    zlist = mod(x1)
    # print(zlist[0].shape)
    # print(zlist[0])
    # print(zlist[0].flatten(0))
    # print(zlist[1].flatten(0))
    # for z in zlist:
    #     print(z)
    #     print(z.shape)
    # print('number is:', len(zlist))