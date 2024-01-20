
import torch
from torch import nn
from torch.nn import init
import time


class SSRNet(nn.Module):
    def __init__(self, stage_num=[3,3,3], image_size=16,
                 class_range=144, lambda_index=1., lambda_delta=1.):
        super(SSRNet, self).__init__()
        self.image_size = image_size
        self.stage_num = stage_num
        self.lambda_index = lambda_index
        self.lambda_delta = lambda_delta
        self.class_range = class_range

        self.stream1_stage3 = nn.Sequential(
            nn.Conv2d(4, 32, 3, 1, 1),
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

        self.stream2_stage3 = nn.Sequential(
            nn.Conv2d(4, 16, 3, 1, 1),
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
        self.funsion_block_stream1_stage_3_before_PB = nn.Sequential(
            nn.Conv2d(32, 10, 1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(8, 8)
        )
        self.funsion_block_stream1_stage_3_prediction_block = nn.Sequential(
            nn.Dropout(0.2, ),
            nn.Linear(10, self.stage_num[2]),
            nn.ReLU()
        )

        self.funsion_block_stream1_stage_2_before_PB = nn.Sequential(
            nn.Conv2d(32, 10, 1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(4, 4)
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
            nn.Linear(10 * 4, self.stage_num[0]),
            nn.ReLU()
        )

        # stream2
        self.funsion_block_stream2_stage_3_before_PB = nn.Sequential(
            nn.Conv2d(16, 10, 1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(8, 8)
        )
        self.funsion_block_stream2_stage_3_prediction_block = nn.Sequential(
            nn.Dropout(0.2, ),
            nn.Linear(10, self.stage_num[2]),
            nn.ReLU()
        )

        self.funsion_block_stream2_stage_2_before_PB = nn.Sequential(
            nn.Conv2d(16, 10, 1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(4, 4)
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
            nn.Linear(10 * 4, self.stage_num[0]),
            nn.ReLU()
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
            nn.Linear(10 * 4, 1),
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

    def forward(self, image_):#[16, 4, 16, 16]
        feature_stream1_stage3 = self.stream1_stage3(image_)#[16, 32, 8, 8]

        feature_stream1_stage2 = self.stream1_stage2(feature_stream1_stage3)#[16, 32, 4, 4]

        feature_stream1_stage1 = self.stream1_stage1(feature_stream1_stage2)#[16, 32, 2, 2]

        feature_stream2_stage3 = self.stream2_stage3(image_)#[16, 16, 8, 8]

        feature_stream2_stage2 = self.stream2_stage2(feature_stream2_stage3)#[16, 16, 4, 4]

        feature_stream2_stage1 = self.stream2_stage1(feature_stream2_stage2)#[16, 16, 2, 2]

        feature_stream1_stage3_before_PB = self.funsion_block_stream1_stage_3_before_PB(feature_stream1_stage3)#[16, 10, 1, 1]
        feature_stream1_stage2_before_PB = self.funsion_block_stream1_stage_2_before_PB(feature_stream1_stage2)#[16, 10, 1, 1]
        feature_stream1_stage1_before_PB = self.funsion_block_stream1_stage_1_before_PB(feature_stream1_stage1)#[16, 10, 2, 2]

        feature_stream2_stage3_before_PB = self.funsion_block_stream2_stage_3_before_PB(feature_stream2_stage3)#[16, 10, 1, 1]
        feature_stream2_stage2_before_PB = self.funsion_block_stream2_stage_2_before_PB(feature_stream2_stage2)#[16, 10, 1, 1]
        feature_stream2_stage1_before_PB = self.funsion_block_stream2_stage_1_before_PB(feature_stream2_stage1)#[16, 10, 2, 2]

        #△k
        embedding_stream1_stage3_before_PB = feature_stream1_stage3_before_PB.view(
            feature_stream1_stage3_before_PB.size(0), -1)#[16, 10]
        embedding_stream1_stage2_before_PB = feature_stream1_stage2_before_PB.view(
            feature_stream1_stage2_before_PB.size(0), -1)#[16, 10]
        embedding_stream1_stage1_before_PB = feature_stream1_stage1_before_PB.view(
            feature_stream1_stage1_before_PB.size(0), -1)#[16, 40]
        embedding_stream2_stage3_before_PB = feature_stream2_stage3_before_PB.view(
            feature_stream2_stage3_before_PB.size(0), -1)#[16, 10]
        embedding_stream2_stage2_before_PB = feature_stream2_stage2_before_PB.view(
            feature_stream2_stage2_before_PB.size(0), -1)#[16, 10]
        embedding_stream2_stage1_before_PB = feature_stream2_stage1_before_PB.view(
            feature_stream2_stage1_before_PB.size(0), -1)#[16, 40]
        stage1_delta_k = self.stage1_delta_k(
            torch.mul(embedding_stream1_stage1_before_PB, embedding_stream2_stage1_before_PB))#[16, 1]
        stage2_delta_k = self.stage2_delta_k(
            torch.mul(embedding_stream1_stage2_before_PB, embedding_stream2_stage2_before_PB))#[16, 1]
        stage3_delta_k = self.stage3_delta_k(
            torch.mul(embedding_stream1_stage3_before_PB, embedding_stream2_stage3_before_PB))#[16, 1]

        embedding_stage1_after_PB = torch.mul(
            self.funsion_block_stream1_stage_1_prediction_block(embedding_stream1_stage1_before_PB),
            self.funsion_block_stream2_stage_1_prediction_block(embedding_stream2_stage1_before_PB))#[16, 3]
        embedding_stage2_after_PB = torch.mul(
            self.funsion_block_stream1_stage_2_prediction_block(embedding_stream1_stage2_before_PB),
            self.funsion_block_stream2_stage_2_prediction_block(embedding_stream2_stage2_before_PB))#[16, 3]
        embedding_stage3_after_PB = torch.mul(
            self.funsion_block_stream1_stage_3_prediction_block(embedding_stream1_stage3_before_PB),
            self.funsion_block_stream2_stage_3_prediction_block(embedding_stream2_stage3_before_PB))#[16, 3]
        embedding_stage1_after_PB = self.stage1_FC_after_PB(embedding_stage1_after_PB)#[16, 6]
        embedding_stage2_after_PB = self.stage2_FC_after_PB(embedding_stage2_after_PB)#[16, 6]
        embedding_stage3_after_PB = self.stage3_FC_after_PB(embedding_stage3_after_PB)#[16, 6]

        prob_stage_1 = self.stage1_prob(embedding_stage1_after_PB)  # [16, 3]
        index_offset_stage1 = self.stage1_index_offsets(embedding_stage1_after_PB)

        prob_stage_2 = self.stage2_prob(embedding_stage2_after_PB)
        index_offset_stage2 = self.stage2_index_offsets(embedding_stage2_after_PB)

        prob_stage_3 = self.stage3_prob(embedding_stage3_after_PB)
        index_offset_stage3 = self.stage3_index_offsets(embedding_stage3_after_PB)

        stage1_regress = prob_stage_1[:, 0] * 0  # [16]
        stage2_regress = prob_stage_2[:, 0] * 0
        stage3_regress = prob_stage_3[:, 0] * 0
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
        regress_class = (stage1_regress + stage2_regress + stage3_regress) * self.class_range  # y=∑ yk * V
        regress_class = torch.squeeze(regress_class, 1)
        return regress_class


def demo_test():
    net = SSRNet()
    # net = net.cuda('cuda')
    # net.eval()
    # x = torch.randn(1, 3, 64, 64).cuda('cuda')
    x = torch.randn(16, 4, 16, 16)
    test_numbers_ = 1000
    a_time = time.time()
    for i in range(test_numbers_):
        y = net(x)
        # print(y.shape)#[16]
    cost_time = time.time() - a_time
    print("time costs:{} s, average_time:{} s\n".format(cost_time, cost_time / test_numbers_))


if __name__ == "__main__":
    demo_test()
