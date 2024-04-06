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
from models import Model
from models import RestNet18
from models import ResNet_50
from models import MobileNetV3_Small
from models import MobileNetV3_Large
from models import SSRNet
# from testGLT import GlobalLocalBrainAge
from randomtest import GlobalLocalBrainAge
from test1 import FusionModel
import utils
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# def softmax(x):
#     x_exp = torch.exp(x)
#     partition = x_exp.sum(1, keepdim=True)
#     return x_exp / partition
def main(args):
    # 初始化模型
    # model = Model(pretrained=True)
    # model = Model(3,1,True)
    # model = Model(4, 1, True)
    # model = Model()
    # model = RestNet18()
    # model = ResNet_50()
    # model = SSRNet()
    model = GlobalLocalBrainAge(3,
                                patch_size=64,
                                nblock=6)
    # model = FusionModel(1)

    # model = MobileNetV3_Small()
    # model = MobileNetV3_Large()

    # model = nn.Sequential(nn.Flatten(), nn.Linear(1024, 1))
    # model = nn.Sequential(nn.Flatten(), nn.Linear(150528, 1))

    model.to(device)
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)

    model.apply(init_weights)

    with open("/home/llj/code/test/model_summary.txt", "w", encoding="utf-8") as f_summary:
        print(model, file=f_summary)

    # 加载权重
    if args.pretrain_weight_path != "":
        state_dict = torch.load(args.pretrain_weight_path, map_location=device)
        model.load_state_dict(state_dict)

    # 损失函数 优化器 学习率调整器
    # criterion = nn.CrossEntropyLoss(reduction='none').to(device)
    # criterion = nn.MSELoss().to(device)#创建一个用于计算损失的均方误差损失函数对象
    criterion = nn.L1Loss().to(device)  # 创建一个用于计算损失的均方误差损失函数对象
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.01)
    # 创建一个AdamW优化器对象，用于更新模型参数。这里采用了AdamW算法，设置了学习率(lr)，权重衰减(weight_decay)和动量(betas)等参数
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=[0.9, 0.999])
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
    # 创建一个学习率调整器对象，用于调整学习率。这里采用了Warmup和Cosine退火的策略，根据训练的步数来调整学习率
    scheduler = utils.WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=int(1.1 * args.epochs))
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

    # 定义数据预处理的操作，包括将图像转换为张量、调整图像大小、应用透视变换和随机旋转等
    transform1 = transforms.Compose([
        # transforms.ColorJitter(contrast=0.8)

        transforms.ToTensor(),
        # transforms.Resize([args.img_size, args.img_size], antialias=True),
        transforms.Resize([170, 120], antialias=True),
        # transforms.Resize([250, 180], antialias=True),
        # transforms.Resize([1000, 720], antialias=True),
        transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
        transforms.RandomRotation(degrees=(0, 180)),
        # transforms.RandomAffine(0, translate=(0.2, 0.2)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip()
    ])
    transform2 = transforms.Compose([
        # transforms.ColorJitter(contrast=0.8)

        transforms.ToTensor(),
        # transforms.Resize([args.img_size, args.img_size], antialias=True),
        transforms.Resize([170, 120], antialias=True),
        # transforms.Resize([250, 180], antialias=True),
        # transforms.Resize([1000, 720], antialias=True),
    ])
    # 创建训练集和验证集的数据集对象，包括图像和标签
    train_dataset = BatchDataset(args.root, args.txt_dir, "train_rb1", transform=transform1)
    val_dataset = BatchDataset(args.root, args.txt_dir, "val_rb1", transform=transform2)
    # 通过数据集对象创建训练集和验证集的数据加载器，用于批量加载数据进行训练和验证
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)

    logging.info('START TIME:{}'.format(time.asctime(time.localtime(time.time()))))
    logging.info(vars(args))
    best_val = None
    meter = utils.ListMeter()
    for epoch in range(args.epochs):
        # 在训练和验证阶段，计算损失和准确率(loss和acc)。并将结果记录在meter对象中
        # 训练
        loss, acc = train(train_loader, model, criterion, optimizer, epoch, args)
        if np.isnan(loss):
            print("ERROR! Loss is Nan. Break.")
            break
        meter.add({"loss": loss, "acc": acc})
        # 验证
        val_loss, val_acc = validate(val_loader, model, criterion, epoch, args)
        meter.add({"val_loss": val_loss, "val_acc": val_acc})
        logging.info(
            "[Epoch:{:<5}/{:<5}] ".format(epoch + 1, args.epochs) +
            "lr:{:.6f} ".format(optimizer.param_groups[0]['lr']) +
            "loss:{:.6f} val_loss:{:.6f} ".format(loss, val_loss) +
            "acc:{:.6f} val_acc:{:.6f}".format(acc, val_acc)
        )
        utils.plot_history(meter.get("loss"), meter.get("acc"), meter.get("val_loss"), meter.get("val_acc"),
                           history_save_path)

        # 保存:将训练和验证的损失和准确率绘制成可视化图表并保存,如果当前的验证损失(val_loss)比之前记录的最佳验证损失(best_val)要低，则保存当前模型的参数到指定路径
        if best_val is None or val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), model_save_path)
            logging.info("Saved best model.")
        scheduler.step()#每个 epoch 结束后，调整学习率(scheduler.step())
    # 最后，绘制整个训练过程中的损失和准确率曲线图，并保存
    utils.plot_history(meter.pop("loss"), meter.pop("acc"), meter.pop("val_loss"), meter.pop("val_acc"),
                       history_save_path)
    logging.info('STOP TIME:{}'.format(time.asctime(time.localtime(time.time()))))

#一个数据加载器(train_loader 或 val_loader)，一个模型(model)，一个损失函数(criterion)，一个优化器(optimizer)，一个epoch数(epoch)，以及其他一些参数
def train(train_loader, model, criterion, optimizer, epoch, args):
    model.train()#将模型设为训练模式
    meter = utils.AverageMeter()
    total = len(train_loader)
    for i, (inputs, times, filenames) in enumerate(train_loader):
        # inputs = torch.reshape(inputs, (-1, 3, 16, 16))
        # inputs = torch.reshape(inputs, (-1, 4, 16, 16))
        # inputs = torch.reshape(inputs, (-1, 6, 16, 16))
        # print(inputs.shape)
        inputs = inputs.to(device)
        # print(inputs)
        # print(inputs.shape)
        times = times.to(device)
        # print(times.shape)
        # print(times)

        time_pd = model(inputs)#预测
        # time_pd = time_pd[0].flatten(0)
        time1 = time_pd[0].flatten(0)
        # # 切片获取除第一个元素以外的所有元素
        # rest_of_list = time_pd[1:]
        # # 计算剩余元素的平均值
        # average = sum(rest_of_list) / len(rest_of_list)
        # time2 = average.flatten(0)

        # loss = criterion(time_pd*144, times*144)
        loss1 = criterion(time1 * 144, times * 144)

        # min_loss = float('inf')
        # for i in range(1, len(time_pd)):
        #     new_list = time_pd.copy()
        #     loss = criterion(new_list[i].flatten(0) * 144, times * 144)
        #     if loss < min_loss:
        #         min_loss = loss
        #         time2 = new_list[i].flatten(0)
        # loss2 = min_loss

        # loss2 = criterion(time2 * 144, times * 144)

        min_loss = float('inf')
        loss2 = 0
        for j in range(1, len(time_pd)):
            new_list = time_pd.copy()
            loss = criterion(new_list[j].flatten(0) * 144, times * 144)
            loss2 += loss
            if loss < min_loss:
                min_loss = loss
                time2 = new_list[j].flatten(0)

        loss = loss1 + loss2
        # loss = min(loss1,loss2)

        # loss = cross_entropy(time_pd, times)
        # acc = utils.accuracy(time_pd, times)
        # acc = torch.eq(torch.ceil(time_pd * 100), torch.ceil(times * 100)).float().cpu().mean()
        # acc = torch.eq(torch.ceil(time_pd), torch.ceil(times)).float().cpu().mean()
        # acc = torch.eq(torch.round(time_pd * 100), torch.round(times * 100)).float().cpu().mean()
        # acc = torch.eq(torch.round(time_pd * 144), torch.round(times * 144)).float().cpu().mean()
        acc1 = torch.eq(torch.round(time1 * 144), torch.round(times * 144)).float().cpu().mean()
        acc2 = torch.eq(torch.round(time2 * 144), torch.round(times * 144)).float().cpu().mean()
        acc = max(acc1, acc2)
        # print(torch.sub(torch.round(time_pd * 100), torch.round(times * 100)))
        # acc = torch.le(abs(torch.sub(torch.round(time_pd * 144), torch.round(times * 144))), 3).float().cpu().mean()
        # acc = torch.le(abs(torch.sub(torch.round(time_pd * 200), torch.round(times * 200))), 6).float().cpu().mean()
        # acc = torch.eq(torch.round(time_pd), torch.round(times)).float().cpu().mean()
        # print(torch.ceil(time_pd))
        # print(torch.ceil(times))
        # print(acc)
        # print(acc.shape)
        # acc = acc1(time_pd, times)
        # acc=0
        meter.add({"loss": float(loss.item()), "acc": acc})
        #如果当前batch的索引(i)被参数args.log_step整除，则打印训练进度(logging.info)，包括当前训练的epoch数、总epoch数、当前batch的索引、总batch数、学习率(optimizer.param_groups[0][‘lr’])以及损失和准确率的信息
        # if i % args.log_step == 0:
        #     logging.info(
        #         "Trainning epoch:{}/{} batch:{}/{} ".format(epoch + 1, args.epochs, i + 1, total) +
        #         "lr:{:.6f} ".format(optimizer.param_groups[0]['lr']) +
        #         "loss:{:.6f} acc:{:.6f}".format(meter.get("loss"), meter.get("acc"))
        #     )
        #接着，将优化器梯度置零(optimizer.zero_grad())，计算总损失(loss)对模型参数的梯度(loss.backward())，并执行一步优化(optimizer.step())更新模型参数
        optimizer.zero_grad()
        # loss.backward()
        loss.backward()
        optimizer.step()

    return meter.pop("loss"), meter.pop("acc")


def validate(val_loader, model, criterion, epoch, args):
    model.eval()
    meter = utils.AverageMeter()
    with torch.no_grad():
        total = len(val_loader)
        for i, (inputs, times, filenames) in enumerate(val_loader):
            # inputs = torch.reshape(inputs, (-1, 3, 16, 16))
            # inputs = torch.reshape(inputs, (-1, 4, 16, 16))
            # inputs = torch.reshape(inputs, (-1, 6, 16, 16))

            inputs = inputs.to(device)
            times = times.to(device)
            time_pd = model(inputs)
            time1 = time_pd[0].flatten(0)

            # rest_of_list = time_pd[1:]
            # average = sum(rest_of_list) / len(rest_of_list)
            # time2 = average.flatten(0)

            # time2 = time_pd[1].flatten(0)

            # time_pd = time_pd[0].flatten(0)
            # time_pd = softmax(time_pd)
            # print(times)
            # print(time_pd)
            # print(time_pd.shape)
            # time_pd = abs(time_pd)

            # loss = criterion(time_pd*144, times*144)
            # loss1 = criterion(time1 * 144, times * 144)
            # loss2 = criterion(time2 * 144, times * 144)

            loss1 = criterion(time1 * 144, times * 144)

            # min_loss = float('inf')
            # for i in range(1, len(time_pd)):
            #     new_list = time_pd.copy()
            #     loss = criterion(new_list[i].flatten(0) * 144, times * 144)
            #     if loss < min_loss:
            #         min_loss = loss
            #         time2 = new_list[i].flatten(0)
            # loss2 = min_loss

            min_loss = float('inf')
            loss2 = 0
            for j in range(1, len(time_pd)):
                new_list = time_pd.copy()
                loss = criterion(new_list[j].flatten(0) * 144, times * 144)
                loss2 += loss
                if loss < min_loss:
                    min_loss = loss
                    time2 = new_list[j].flatten(0)

            # loss2 = criterion(time2 * 144, times * 144)
            loss = loss1 + loss2
            # loss = min(loss1, loss2)
            # acc = utils.accuracy(time_pd, times)
            # acc = torch.eq(time_pd, times).float().mean()
            # acc = torch.eq(torch.ceil(time_pd), torch.ceil(times)).float().cpu().mean()
            # acc = torch.eq(torch.round(time_pd*100), torch.round(times*100)).float().cpu().mean()
            # acc = torch.le(abs(torch.sub(torch.round(time_pd * 100), torch.round(times * 100))), 6).float().cpu().mean()
            # acc = torch.le(abs(torch.sub(torch.round(time_pd * 144), torch.round(times * 144))), 3).float().cpu().mean()
            # acc = torch.eq(torch.round(time_pd * 144), torch.round(times * 144)).float().cpu().mean()
            acc1 = torch.eq(torch.round(time1 * 144), torch.round(times * 144)).float().cpu().mean()
            acc2 = torch.eq(torch.round(time2 * 144), torch.round(times * 144)).float().cpu().mean()
            acc = max(acc1, acc2)
            # acc = torch.eq(torch.round(time_pd), torch.round(times)).float().cpu().mean()
            # acc = acc1(time_pd, times)
            meter.add({"loss": loss.item(), "acc": acc})


            # if i % args.log_step == 0:
            #     logging.info(
            #         "Validating epoch:{}/{} batch:{}/{} ".format(epoch + 1, args.epochs, i + 1, total) +
            #         "loss:{:.6f} acc:{:.6f}".format(meter.get("loss"), meter.get("acc"))
            #     )

    return meter.pop("loss"), meter.pop("acc")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="")#创建一个ArgumentParser对象，用于解析命令行参数
    parser.add_argument("--root", type=str, default="/home/llj/code/test/data_rb")#添加一个命令行参数--root，类型为字符串，必需参数
    parser.add_argument("--txt_dir", type=str, default="/home/llj/code/test/")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_step", type=int, default=100)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--pretrain_weight_path", type=str, default="", help="pretrain weight path")
    parser.add_argument("--experiment_name", type=str, default="llj",help="experiment name")
    args = parser.parse_args()#解析命令行参数，并将结果存储在args变量中
    # 根据实验名称和当前时间生成模型保存路径
    model_save_path = "./middle/models/{}-{}-best.pth".format(args.experiment_name,
                                                              datetime.now().strftime("%Y%m%d-%H%M%S"))
    # 根据实验名称和当前时间生成日志文件路径
    log_path = "./middle/logs/{}-{}.log".format(args.experiment_name, datetime.now().strftime("%Y%m%d-%H%M%S"))
    # 根据实验名称和当前时间生成历史记录保存路径
    history_save_path = "./middle/history/{}-{}.png".format(args.experiment_name,
                                                            datetime.now().strftime("%Y%m%d-%H%M%S"))

    os.makedirs("./middle/models/", exist_ok=True)# 创建存储模型文件的目录
    os.makedirs("./middle/logs/", exist_ok=True)
    os.makedirs("./middle/history/", exist_ok=True)
    # 配置日志记录的格式和处理器
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_path, mode='a'), logging.StreamHandler()]
    )
    try:
        device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
        utils.fix_seed()#调用名为fix_seed()的函数，用于设置随机种子，保证实验的可复现性
        main(args)
    except Exception as e:
        logging.error(e)
        logging.error(traceback.format_exc())
        sys.exit()