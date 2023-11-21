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
from models import RestNet18
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def eval():

    # 加载模型
    # model = Model()
    # model = Model(3, 1, True)
    # model = nn.Sequential(nn.Flatten(), nn.Linear(256, 1))
    # model.to(device)
    model = RestNet18()
    model.to(device)
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)

    model.apply(init_weights)
    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()


    # 加载数据
    transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Resize([args.img_size, args.img_size], antialias=True),
    ])
    dataset = BatchDataset(args.root, args.txt_dir, "train2", transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.num_workers, pin_memory=True)

    # 保存结果
    fp = open(result_save_path, "w", encoding="utf-8-sig")
    fp.write("filename,time(gt),time(pd)\n")
    with torch.no_grad():
        total = len(loader)
        for i, (inputs, times, filenames) in enumerate(loader):
            print("{:<5}/{:<5}".format(i, total), end="\r")
            inputs = torch.reshape(inputs, (-1, 3, 16, 16))
            inputs = inputs.to(device)
            # print(inputs)
            times = times.to(device)
            # print(times)
            time_pd = model(inputs)
            # time_pd = abs(time_pd)
            # print(time_pd)

            for j in range(inputs.shape[0]):
                fp.write("{},{},{}\n".format(
                    filenames[j], round(times[j].item()*100),
                    round(time_pd[j].item()*100)
                ))

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
            filename, time_gt,time_pd= line.strip().split(",")
            loss += (int(time_gt)/100.0 - int(time_pd)/100.0)**2
            # loss += (float(time_gt) / 100.0 - float(time_pd) / 100.0) ** 2
            if int(time_gt) == int(time_pd) :
            # if (abs(float(time_gt) - float(time_pd))<4):
            # time_gt = int(time_gt)
            # time_pd = int(time_pd)
            # if Acc(time_gt, time_pd):
                count +=1
    print("loss:{}, acc:{:.6f}".format(loss/total, count/total))  # 此处离线评估的loss会比训练期间的验证集更小 因为保存csv时用round做了四舍五入取整
    # print("loss:{}".format(loss / total))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--root", type=str, default="/home/llj/code/test/data2")
    parser.add_argument("--txt_dir", type=str, default="/home/llj/code/test/")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--weights", type=str, default="/home/llj/code/test/middle/models/llj-20231120-201959-best.pth", help="pretrain weight path")
    parser.add_argument("--experiment_name", type=str, default="llj",help="experiment name")
    # parser.add_argument("--mode", type=str, required=True, choices=["eval", "metrics"])
    parser.add_argument("--mode", type=str, default="eval")
    args = parser.parse_args()

    result_save_path = "./middle/result/{}2.csv".format(args.experiment_name)

    os.makedirs("./middle/result/", exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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