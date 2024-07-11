# -*- "coding: utf-8" -*-
import torch.nn as nn
import os
import argparse
import torch
import torch.utils.data
import torchvision.transforms as transforms

from main import BatchDataset
from models import Model
import utils
from randomtest import GlobalLocalBrainAge
from test1 import FusionModel
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def eval():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = GlobalLocalBrainAge(3,
                                patch_size=64,
                                nblock=6)
    model.to(device)

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)
    model.apply(init_weights)
    state_dict = torch.load("./middle/models/temp_model2.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    # 加载数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([170, 120], antialias=True),
        # transforms.Resize([120, 160], antialias=True),
        # transforms.Resize([120, 170], antialias=True),
        # transforms.Resize([128, 170], antialias=True),
    ])
    dataset = BatchDataset("", "", "all2_temp2", transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False,num_workers=4, pin_memory=True)

    # 保存结果
    # fp = open("./results/ps96_temp2_avgall.csv", "w", encoding="utf-8-sig")
    # fp2 = open("./results/ps96_temp2_avg.csv", "w", encoding="utf-8-sig")
    fp = open("./results/temp_model2.csv", "w", encoding="utf-8-sig")
    # fp2 = open("./results/105_avg.csv", "w", encoding="utf-8-sig")
    # fp2.write("filename,time(gt),time(pd)\n")
    with torch.no_grad():
        total = len(loader)
        for i, (inputs, times, filenames) in enumerate(loader):
            # print("{:<5}/{:<5}".format(i, total), end="\r")
            inputs = inputs.to(device)
            times = times.to(device)
            time_pd = model(inputs)
            #取最好
            # for j in range(inputs.shape[0]):
            #     time_init = times[j].item()*144
            #     time_best = time_pd[0].flatten(0)
            #     time_pd_best = time_best[j].item()*144
            #     for k in range(1, len(time_pd)):
            #         time2 = time_pd[k].flatten(0)
            #         time2_pd = time2[j].item() * 144
            #         if abs(time_init - time2_pd) < abs(time_init - time_pd_best):
            #             time_pd_best = time2_pd
            #     fp.write("{},{},{}\n".format(
            #         filenames[j], round(time_init),
            #         round(time_pd_best)
            #     ))

            #局部平均
            for j in range(inputs.shape[0]):
                time_init = times[j].item() * 24 +38
                total_time = 0  # 用于累积time[j]的总和
                count = 0  # 用于计数time[j]的数量
                for k in range(1, len(time_pd)):
                    time2 = time_pd[k].flatten(0)
                    time2_pd = time2[j].item() * 24+38
                    total_time += time2_pd  # 累积time[j]的总和
                    count += 1  # 增加time[j]的数量
                average_time_j = total_time / count  # 计算time[j]的平均值
                fp.write("{},{},{}\n".format(
                    filenames[j], round(time_init,1),
                    round(average_time_j,1)  # 写入平均值
                ))

            # # 全局一起平均和只有局部平均
            # for j in range(inputs.shape[0]):
            #     time_init = times[j].item() * 24 +38
            #     # time_init = times[j].item() * 4 + 37
            #     # time_init = times[j].item() * 144
            #     total_time = 0  # 用于累积time[j]的总和
            #     count = 0  # 用于计数time[j]的数量
            #     time1 = time_pd[0].flatten(0)
            #     time1_pd = time1[j].item() * 24+38
            #     # time1_pd = time1[j].item() * 4 + 37
            #     # time1_pd = time1[j].item() * 144
            #     avg1 = time1_pd
            #     for k in range(1, len(time_pd)):
            #         time2 = time_pd[k].flatten(0)
            #         time2_pd = time2[j].item() * 24+38
            #         # time2_pd = time2[j].item() * 4 + 37
            #         # time2_pd = time2[j].item() * 144
            #         total_time += time2_pd  # 累积time[j]的总和
            #         count += 1  # 增加time[j]的数量
            #     average_time_j = total_time / count  # 计算time[j]的平均值
            #     avg1 += total_time
            #     cn = count + 1
            #     avg1 = avg1 / cn
            #     #有全局
            #     fp.write("{},{},{}\n".format(
            #         filenames[j], round(time_init,1),
            #         round(avg1,1)
            #     ))
            #     #无全局
            #     fp2.write("{},{},{}\n".format(
            #         filenames[j], round(time_init,1),
            #         round(average_time_j,1)  # 写入平均值
            #     ))

            # # #加权平均
            # global_weight = 0.2
            # local_weight = 0.8
            # time1 = time_pd[0].flatten(0)
            # weighted_average_result = (time1 * global_weight)  # 初始化加权平均结果为全局结果
            # local_weight_sum = 0  # 初始化局部结果的权重总和
            # for j in range(1, len(time_pd)):
            #     time_local = time_pd[j].flatten(0)
            #     weighted_average_result += time_local * local_weight
            #     local_weight_sum += local_weight
            # # 考虑所有局部结果的加权平均
            # weighted_average_result /= (global_weight + local_weight_sum)
            # for j in range(inputs.shape[0]):
            #     time_init = times[j].item() * 144
            #     fp2.write("{},{},{}\n".format(
            #         filenames[j], round(time_init),
            #         round(weighted_average_result[j].item() * 144)  # 写入加权平均结果
            #     ))

            # for j in range(inputs.shape[0]):
            #     time_init = times[j].item()*144
            #     time_best = time_pd[0].flatten(0)
            #     time_pd_best = time_best[j].item()*144
            #     total_time = 0
            #     count = 0
            #     for k in range(1, len(time_pd)):
            #         time2 = time_pd[k].flatten(0)
            #         time2_pd = time2[j].item() * 144
            #         total_time += time2_pd
            #         count += 1
            #         if abs(time_init - time2_pd) < abs(time_init - time_pd_best):
            #             time_pd_best = time2_pd
            #     average_time_j = total_time / count
            #     fp.write("{},{},{}\n".format(
            #         filenames[j], round(time_init),
            #         round(time_pd_best)
            #     ))
            #     fp2.write("{},{},{}\n".format(
            #         filenames[j], round(time_init),
            #         round(average_time_j)
            #     ))
    metrics()
def metrics():
    # count = 0
    # total = 0
    # loss = 0
    # with open("./results/ps96_temp2_avgall.csv", "r", encoding="utf-8-sig") as f:
    # # with open("./results/105_avg.csv", "r", encoding="utf-8-sig") as f:
    #     f.readline()
    #     for line in f:
    #         total += 1
    #         filename, time_gt, time_pd = line.strip().split(",")
    #         # loss += abs(int(time_gt) - int(time_pd))
    #         # if abs(int(time_gt) - int(time_pd)) <= 4:
    #         loss += abs(float(time_gt) - float(time_pd))
    #         if abs(float(time_gt) - float(time_pd)) <= 4:
    #             count += 1
    # with open("loss_acc_ps96_temp2_avgall.txt", 'a', encoding="utf-8-sig") as f:
    # # with open("loss_acc45_avg.txt", 'a', encoding="utf-8-sig") as f:
    #     f.write("{},{}\n".format(
    #         loss/total, count/total)
    #     )
    count2 = 0
    total2 = 0
    loss2 = 0
    with open("./results/temp_model2.csv", "r", encoding="utf-8-sig") as f:
    # with open("./results/105_avgall.csv", "r", encoding="utf-8-sig") as f:
        f.readline()
        for line in f:
            total2 += 1
            filename, time_gt, time_pd = line.strip().split(",")
            # loss2 += abs(int(time_gt) - int(time_pd))
            # if abs(int(time_gt) - int(time_pd)) <= 4:
            loss2 += abs(float(time_gt) - float(time_pd))
            if abs(float(time_gt) - float(time_pd)) <= 1:
                count2 += 1
    with open("loss_acc_temp_model2.txt", 'a', encoding="utf-8-sig") as f:
    # with open("loss_acc45_avgall.txt", 'a', encoding="utf-8-sig") as f:
        f.write("{},{}\n".format(
            loss2/total2, count2/total2)
        )
    # print("loss:{}, acc:{:.6f}".format(loss/total, count/total))  # 此处离线评估的loss会比训练期间的验证集更小 因为保存csv时用round做了四舍五入取整
def main_eval():
    utils.fix_seed()
    eval()