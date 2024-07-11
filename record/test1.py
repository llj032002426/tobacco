import torch
from torch import nn
from torch.nn import init
import math
class convBlock(nn.Module):
    def __init__(self, inplace, outplace, kernel_size=3, padding=1):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplace, outplace, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(outplace)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x


class VGG8(nn.Module):
    def __init__(self, inplace):
        super().__init__()

        ly = [64, 128, 256, 512]

        self.ly = ly

        self.maxp = nn.MaxPool2d(2)

        self.conv11 = convBlock(inplace, ly[0])
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


class GlobalAttention(nn.Module):
    def __init__(self,
                 transformer_num_heads=8,
                 hidden_size=512,
                 transformer_dropout_rate=0.0):
        super().__init__()

        self.num_attention_heads = transformer_num_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(transformer_dropout_rate)
        self.proj_dropout = nn.Dropout(transformer_dropout_rate)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, locx, glox):
        locx_query_mix = self.query(locx)
        glox_key_mix = self.key(glox)
        glox_value_mix = self.value(glox)

        query_layer = self.transpose_for_scores(locx_query_mix)
        key_layer = self.transpose_for_scores(glox_key_mix)
        value_layer = self.transpose_for_scores(glox_value_mix)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)

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


class GlobalLocalBrainAge(nn.Module):
    def __init__(self, inplace,
                 patch_size=64,
                 step=-1,
                 nblock=6,
                 drop_rate=0.5,
                 backbone='vgg8'):
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

        if backbone == 'vgg8':
            self.global_feat = VGG8(inplace)
            self.local_feat = VGG8(inplace)
            hidden_size = 512
        elif backbone == 'vgg16':
            self.global_feat = VGG16(inplace)
            self.local_feat = VGG16(inplace)
            hidden_size = 512
        else:
            raise ValueError('% model does not supported!' % backbone)

        self.attnlist = nn.ModuleList()
        self.fftlist = nn.ModuleList()

        for n in range(nblock):
            atten = GlobalAttention(
                transformer_num_heads=8,
                hidden_size=hidden_size,
                transformer_dropout_rate=drop_rate)
            self.attnlist.append(atten)

            fft = Feedforward(inplace=hidden_size * 2,
                              outplace=hidden_size)
            self.fftlist.append(fft)

        self.avg = nn.AdaptiveAvgPool2d(1)
        out_hidden_size = hidden_size

        self.gloout = nn.Linear(out_hidden_size, 1)
        self.locout = nn.Linear(out_hidden_size, 1)

    def forward(self, xinput):
        _, _, H, W = xinput.size()
        outlist = []

        xglo = self.global_feat(xinput)
        xgfeat = torch.flatten(self.avg(xglo), 1)

        glo = self.gloout(xgfeat)
        outlist = [glo]

        B2, C2, H2, W2 = xglo.size()
        xglot = xglo.view(B2, C2, H2 * W2)
        xglot = xglot.permute(0, 2, 1)

        for y in range(0, H - self.patch_size, self.step):
            for x in range(0, W - self.patch_size, self.step):
                locx = xinput[:, :, y:y + self.patch_size, x:x + self.patch_size]
                xloc = self.local_feat(locx)

                for n in range(self.nblock):
                    B1, C1, H1, W1 = xloc.size()
                    xloct = xloc.view(B1, C1, H1 * W1)
                    xloct = xloct.permute(0, 2, 1)

                    tmp = self.attnlist[n](xloct, xglot)
                    tmp = tmp.permute(0, 2, 1)
                    tmp = tmp.view(B1, C1, H1, W1)
                    tmp = torch.cat([tmp, xloc], 1)

                    tmp = self.fftlist[n](tmp)
                    xloc = xloc + tmp

                xloc = torch.flatten(self.avg(xloc), 1)

                out = self.locout(xloc)
                outlist.append(out)

        # return outlist[0].flatten(0)
        return outlist[0]


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
        # regress_class = torch.squeeze(regress_class, 1)
        return regress_class

# 多层级融合：将SSRnet和Global-Local Transformer模型的多个层级的特征进行融合，而不仅仅是最后一层的输出。这样可以利用更多层级的特征信息，提高融合的效果
class FusionModel(nn.Module):
    def __init__(self, num_classes):
        super(FusionModel, self).__init__()
        self.ssrnet = SSRNet()
        self.gltransformer = GlobalLocalBrainAge(4,
                              patch_size=64,
                              step=32,
                              nblock=6,
                              backbone='vgg8')
        self.fc = nn.Linear(2 * num_classes, num_classes)

    def forward(self, x):
        ssr_output = self.ssrnet(x) #[16,1]
        gl_output = self.gltransformer(x) #[16,1]
        combined_output = torch.cat((ssr_output, gl_output), dim=1) #[16, 2]
        output = self.fc(combined_output) #[16, 1]
        return output.flatten(0)

if __name__ == '__main__':
    # x1 = torch.rand(1, 5, 130, 170)
    x1 = torch.rand(16, 4, 16, 16)

    model = FusionModel(1)
    x = model(x1)
    print(x.shape)
