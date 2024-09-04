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
from models import ResNet18
from models import ResNet_50
from models import SSRNet
from models import VGG16
# from testGLT import GlobalLocalBrainAge
from randomtest import GlobalLocalBrainAge
from test1 import FusionModel
from model_compare import ghostnet

def eval():
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 加载模型
    # model = Model()
    # model = Model(3, 1, True)
    # model = nn.Sequential(nn.Flatten(), nn.Linear(256, 1))
    # model.to(device)
    # model = ResNet18()
    # model = VGG16()
    # model = ResNet_50()
    # model = SSRNet()
    model = GlobalLocalBrainAge(3,
                                patch_size=96,
                                nblock=6)
    # model = FusionModel(1)
    # model = ghostnet()
    model.to(device)
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)

    # def init_weights(m):
    #     if type(m) == nn.Linear:
    #         nn.init.xavier_normal_(m.weight)

    # def init_weights(m):
    #     if type(m) == nn.Linear:
    #         nn.init.kaiming_normal_(m.weight)
    model.apply(init_weights)
    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()


    # 加载数据
    transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Resize([args.img_size, args.img_size], antialias=True),
                # transforms.Resize([120, 170], antialias=True),
                transforms.Resize([170, 120], antialias=True),
    ])
    dataset = BatchDataset(args.root, args.txt_dir, "all2_hum", transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.num_workers, pin_memory=True)

    fp = open(result_save_path, "w", encoding="utf-8-sig")
    # fp.write("filename,time(gt),time(pd)\n")
    with torch.no_grad():
        total = len(loader)
        for i, (inputs, times, filenames) in enumerate(loader):
            print("{:<5}/{:<5}".format(i, total), end="\r")
            # inputs = torch.reshape(inputs, (-1, 3, 16, 16))
            # inputs = torch.reshape(inputs, (-1, 4, 16, 16))
            # inputs = torch.reshape(inputs, (-1, 6, 16, 16))
            inputs = inputs.to(device)
            # print(inputs)
            times = times.to(device)
            # print(times)
            time_pd = model(inputs)
            # time1 = time_pd[0].flatten(0)

            # rest_of_list = time_pd[1:]
            # average = sum(rest_of_list) / len(rest_of_list)
            # time2 = average.flatten(0)

            for j in range(inputs.shape[0]):
                # time_init = times[j].item() * 144
                # time_init = times[j].item() * 24 + 38
                time_init = times[j].item() * 4 + 37
                total_time = 0  # 用于累积time[j]的总和
                count = 0  # 用于计数time[j]的数量
                for k in range(1, len(time_pd)):
                    time2 = time_pd[k].flatten(0)
                    # time2_pd = time2[j].item() * 144
                    # time2_pd = time2[j].item() * 24 + 38
                    time2_pd = time2[j].item() * 4 + 37
                    total_time += time2_pd  # 累积time[j]的总和
                    count += 1  # 增加time[j]的数量
                average_time_j = total_time / count  # 计算time[j]的平均值
                # fp.write("{},{},{}\n".format(
                #     filenames[j], round(time_init,1),
                #     round(average_time_j,1)  # 写入平均值
                # ))
                # fp.write("{},{},{}\n".format(
                #     filenames[j], round(time_init,1),
                #     round(average_time_j,1)  # 写入平均值
                # ))
                fp.write("{},{},{}\n".format(
                    filenames[j], round(time_init,1),
                    round(average_time_j,1)  # 写入平均值
                ))

            # # 普通
            # for j in range(inputs.shape[0]):
            #     # time_init = times[j].item() * 24 +38
            #     time_init = times[j].item() * 4 + 37
            #     time2 = time_pd.flatten(0)
            #     # time2_pd = time2[j].item() * 24+38
            #     time2_pd = time2[j].item() * 4 + 37
            #     fp.write("{},{},{}\n".format(
            #         filenames[j], round(time_init,1),
            #         round(time2_pd,1)  # 写入平均值
            #     ))

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

            # for j in range(inputs.shape[0]):
            #     # time_init = times[j].item()*144
            #     # time1_pd = time1[j].item()*144
            #     # time2_pd = time2[j].item() * 144
            #     # if abs(time_init - time1_pd) < abs(time_init - time2_pd):
            #     #     time_pd = time1_pd
            #     # else:
            #     #     time_pd = time2_pd
            #     # fp.write("{},{},{}\n".format(
            #     #     filenames[j], round(time_init),
            #     #     round(time_pd)
            #     # ))
            #     fp.write("{},{},{}\n".format(
            #         filenames[j], round(times[j].item() * 144),
            #         round(time_pd[j].item() * 144)
            #     ))

        print("{:<5}/{:<5}".format(i, total))
    metrics()

def Acc(t1,t2):
    a1 = int(t1 / 24)
    b1 = t1 % 24
    a2 = int(t2 / 24)
    b2 = t2 % 24
    if a1 == a2 and int(b1 / 6) == int(b2 / 6):
        return True
    return False

def metrics():
    count = 0
    total = 0
    loss = 0
    with open(result_save_path, "r", encoding="utf-8-sig") as f:
        f.readline()
        for line in f:
            total += 1
            filename, time_gt, time_pd = line.strip().split(",")
            # loss += (int(time_gt)/100.0 - int(time_pd)/100.0)**2
            # loss += (int(time_gt) - int(time_pd)) ** 2
            # loss += (float(time_gt) / 100.0 - float(time_pd) / 100.0) ** 2
            # if int(time_gt) == int(time_pd) :
            # loss += abs(int(time_gt) - int(time_pd))
            # if abs(int(time_gt) - int(time_pd)) <= 4:
            loss += abs(float(time_gt) - float(time_pd))
            if abs(float(time_gt) - float(time_pd)) <= 1:
                count += 1
    # with open("loss_acc.txt", "w", encoding="utf-8-sig") as f:
    #     f.write("{},{}\n".format(
    #         loss/total, count/total )
    #     )
    print("loss:{}, acc:{:.6f}".format(loss/total, count/total))  # 此处离线评估的loss会比训练期间的验证集更小 因为保存csv时用round做了四舍五入取整
    # print("loss:{}".format(loss / total))

if __name__ == '__main__':
# def main2():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--root", type=str, default="/home/ubuntu/llj/tobacco/data2")
    parser.add_argument("--txt_dir", type=str, default="/home/ubuntu/llj/tobacco/")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--weights", type=str, default="./middle/models/hum_ps96_n6_noFFT-best (1).pth", help="pretrain weight path")
    parser.add_argument("--experiment_name", type=str, default="llj",help="experiment name")
    # parser.add_argument("--mode", type=str, required=True, choices=["eval", "metrics"])
    parser.add_argument("--mode", type=str, default="eval")
    args = parser.parse_args()

    # result_save_path = "./middle/result/{}2.csv".format(args.experiment_name)
    result_save_path = "./results/hum_ps96_n6_noFFT_1.csv".format(args.experiment_name)
    # result_save_path = "hum_AlexNet_3.csv".format(args.experiment_name)

    os.makedirs("./middle/result/", exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    utils.fix_seed()
    if args.mode == "eval":
        assert os.path.exists(args.root), f"Dataset path '{args.root}' NOT exists."
        assert os.path.exists(args.txt_dir), f"*.txt path '{args.txt_dir}' NOT exists."
        # assert os.path.exists(args.weights), f"Weights path '{args.weights}' NOT exists."
        # with open(os.path.join(args.txt_dir, "classes.txt"), "r", encoding="utf-8") as f:
        #     names = f.read().strip().split("\n")
        eval()
    elif args.mode == "metrics":
        assert os.path.exists(result_save_path), f"CSV path '{result_save_path}' NOT exists."
        metrics()
    else:
        print("Invalid mode:{}".format(args.mode))