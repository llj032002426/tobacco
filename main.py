import glob
import numpy
import torch
from sklearn.preprocessing import MinMaxScaler
import cv2
import os
import random
import numpy as np
import scipy.io as sio
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from group_feature import Group_2G_R_B,Group_GB,Group_RB,Group_GR,Group_lab,Group_RBimg
from textural_feature import fast_glcm_mean,fast_glcm_std,fast_glcm_contrast,fast_glcm_dissimilarity,fast_glcm_homogeneity,fast_glcm_ASM,fast_glcm_ENE,fast_glcm_max,fast_glcm_entropy,all_glcm
from PIL import  Image
import os
import numpy as np
import torchvision.transforms as transforms
import datetime
import time

def generate_all_txt(root, txt_dir, num_samples=None):
    """ 筛选符合要求的数据并生成 all.txt 文件 """
    f = open(os.path.join(txt_dir, "all_rb.txt"), "w", encoding="utf-8")
    all_imgs_path = glob.glob(r'/home/llj/code/test/data_rb/*/*.jpg')#数据文件夹路径
    for var in all_imgs_path:
        # print(var)
        file_path, file_name = os.path.split(var)
        parent_path, parent_name = os.path.split(file_path)
        # print(parent_name,file_name)
        parent_name=parent_name[:4]+'-'+parent_name[4:6]+'-'+parent_name[6:]
        a_s=tuple(time.strptime(parent_name, "%Y-%m-%d"))
        d1 = datetime.date(a_s[0], a_s[1], a_s[2])
        start = '20230610'
        start = start[:4] + '-' + start[4:6] + '-' + start[6:]
        s = tuple(time.strptime(start, "%Y-%m-%d"))
        d2 = datetime.date(s[0], s[1], s[2])
        # times=(d1 - d2).days*24+int(file_name[:2])+round(int(file_name[2:4])/60,3)
        times = (d1 - d2).days * 24 + int(file_name[:2])-17

        # times = int(times / 6)
        # print(times)
        f.write(f"{var},{times}\n")

def generate_all_txt2(root, txt_dir, num_samples=None):
    """ 筛选符合要求的数据并生成 all.txt 文件 """
    f = open(os.path.join(txt_dir, "all2_rb.txt"), "w", encoding="utf-8")
    all_imgs_path = glob.glob(r'/home/llj/code/test/data2_rb/*/*.jpg')#数据文件夹路径
    for var in all_imgs_path:
        # print(var)
        file_path, file_name = os.path.split(var)
        parent_path, parent_name = os.path.split(file_path)
        # print(parent_name,file_name)
        parent_name=parent_name[:4]+'-'+parent_name[4:6]+'-'+parent_name[6:]
        a_s=tuple(time.strptime(parent_name, "%Y-%m-%d"))
        d1 = datetime.date(a_s[0], a_s[1], a_s[2])
        start = '20230625'
        start = start[:4] + '-' + start[4:6] + '-' + start[6:]
        s = tuple(time.strptime(start, "%Y-%m-%d"))
        d2 = datetime.date(s[0], s[1], s[2])
        # times=(d1 - d2).days*24+int(file_name[:2])+round(int(file_name[2:4])/60,3)
        times = (d1 - d2).days * 24 + int(file_name[:2])-11

        # times = int(times / 6)
        # print(times)
        f.write(f"{var},{times}\n")

def split_train_test(txt_dir, ratio=0.7, seed=123):
    '''拆分训练集和测试集
    '''
    with open(os.path.join(txt_dir, "all_rb.txt"), "r", encoding="utf-8") as f:
        lines = f.readlines()
    total = len(lines)

    random.seed(seed)
    random.shuffle(lines)
    n_train = int(total * ratio)
    train_lines = lines[:n_train]
    val_lines = lines[n_train:]

    with open(os.path.join(txt_dir, "train_rb1.txt"), "w", encoding="utf-8") as f:
        for line in train_lines:
            f.write(line)
    with open(os.path.join(txt_dir, "val_rb1.txt"), "w", encoding="utf-8") as f:
        for line in val_lines:
            f.write(line)

import os
import random
def split_data(txt_dir, train_ratio=0.6, val_ratio=0.2, seed=123):
    """将数据集按照比例分为训练集、验证集和测试集三个"""
    with open(os.path.join(txt_dir, "all_rb.txt"), "r", encoding="utf-8") as f:
        lines = f.readlines()
    total = len(lines)

    random.seed(seed)
    random.shuffle(lines)

    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size

    train_lines = lines[:train_size]
    val_lines = lines[train_size:train_size + val_size]
    test_lines = lines[-test_size:]

    with open(os.path.join(txt_dir, "train_rb2.txt"), "w", encoding="utf-8") as f:
        for line in train_lines:
            f.write(line)
    with open(os.path.join(txt_dir, "val_rb2.txt"), "w", encoding="utf-8") as f:
        for line in val_lines:
            f.write(line)
    with open(os.path.join(txt_dir, "test_rb2.txt"), "w", encoding="utf-8") as f:
        for line in test_lines:
            f.write(line)

def split_data2(txt_dir, train_ratio=0.6, val_ratio=0.2, seed=123):
    """将数据集按照比例分为训练集、验证集和测试集三个"""
    with open(os.path.join(txt_dir, "all2.txt"), "r", encoding="utf-8") as f:
        lines = f.readlines()
    total = len(lines)

    random.seed(seed)
    random.shuffle(lines)

    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size

    train_lines = lines[:train_size]
    val_lines = lines[train_size:train_size + val_size]
    test_lines = lines[-test_size:]

    with open(os.path.join(txt_dir, "train2.txt"), "w", encoding="utf-8") as f:
        for line in train_lines:
            f.write(line)
    with open(os.path.join(txt_dir, "val2.txt"), "w", encoding="utf-8") as f:
        for line in val_lines:
            f.write(line)
    with open(os.path.join(txt_dir, "test2.txt"), "w", encoding="utf-8") as f:
        for line in test_lines:
            f.write(line)


def split_train_test2(txt_dir, ratio=0.6, seed=123):
    '''拆分训练集和测试集
    '''
    with open(os.path.join(txt_dir, "all2.txt"), "r", encoding="utf-8") as f:
        lines = f.readlines()
    total = len(lines)

    random.seed(seed)
    random.shuffle(lines)
    n_train = int(total * ratio)
    train_lines = lines[:n_train]
    val_lines = lines[n_train:]

    with open(os.path.join(txt_dir, "train2.txt"), "w", encoding="utf-8") as f:
        for line in train_lines:
            f.write(line)
    with open(os.path.join(txt_dir, "val2.txt"), "w", encoding="utf-8") as f:
        for line in val_lines:
            f.write(line)

class BatchDataset(Dataset):

    def __init__(self, root, txt_dir, name, transform):
        # self.root = root
        self.transform = transform
        with open(os.path.join(txt_dir, f"{name}.txt"), "r", encoding="utf-8") as f:
            self.lines = f.readlines()

    def __getitem__(self, idx):
        filename, times = self.lines[idx].strip().split(",")
        # src = cv2.imread(filename)
        # image = Group_RBimg(src)
        image = Image.open(filename).convert('RGB')
        image = self.transform(image)
        times = np.float32(int(times) / 144.0)
        return (image, times, filename)

        # times = np.float32(int(times) / 144.0)
        # img = np.array(Image.open(filename).resize((160, 120)).convert('L'))
        # src = cv2.imread(filename)
        # # src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)  # CV BGR转变RGB
        # # x = 500
        # # y = 500
        # # w = 1000
        # # h = 1500
        # # src = src[x:x + w, y:y + h]
        # src = Group_RBimg(src)
        # hist_0 = cv2.calcHist([src], [0], None, [256], [0, 256])
        # hist_1 = cv2.calcHist([src], [1], None, [256], [0, 256])
        # hist_2 = cv2.calcHist([src], [2], None, [256], [0, 256])
        # hist_0 = torch.Tensor(hist_0)
        # hist_1 = torch.Tensor(hist_1)
        # hist_2 = torch.Tensor(hist_2)
        # hist = torch.stack((hist_0, hist_1, hist_2), 0)
        # # hist2 = Group_lab(src)
        # # hist = torch.cat((hist, hist2), dim=0)
        # # img = np.array(Image.fromarray(image_array).resize((160, 120)).convert('L'))
        # # texture_feat = fast_glcm_mean(img)
        # # texture_feat = fast_glcm_std(img)
        # texture_feat1 = fast_glcm_contrast(img)
        # # texture_feat = fast_glcm_dissimilarity(img)
        # # texture_feat = fast_glcm_homogeneity(img)
        # texture_feat2 = fast_glcm_ASM(img)
        # # texture_feat = fast_glcm_ENE(img)
        # # texture_feat = fast_glcm_max(img)
        # texture_feat3 = fast_glcm_entropy(img)
        # # texture_feat = all_glcm(img)
        # texture_feat1 = torch.Tensor(texture_feat1)
        # texture_feat2 = torch.Tensor(texture_feat2)
        # texture_feat3 = torch.Tensor(texture_feat3)
        # texture_feat1 = torch.unsqueeze(texture_feat1,0)
        # texture_feat1 = torch.unsqueeze(texture_feat1, 0)
        # texture_feat2 = torch.unsqueeze(texture_feat2, 0)
        # texture_feat2 = torch.unsqueeze(texture_feat2, 0)
        # texture_feat3 = torch.unsqueeze(texture_feat3, 0)
        # texture_feat3 = torch.unsqueeze(texture_feat3, 0)
        # texture_feat1 = torch.nn.functional.interpolate(texture_feat1, size=(256, 1), mode='bilinear',
        #                                                           align_corners=False)
        # texture_feat2 = torch.nn.functional.interpolate(texture_feat2, size=(256, 1), mode='bilinear',
        #                                                align_corners=False)
        # texture_feat3 = torch.nn.functional.interpolate(texture_feat3, size=(256, 1), mode='bilinear',
        #                                                align_corners=False)
        # texture_feat1 = torch.squeeze(texture_feat1, 0)
        # texture_feat2 = torch.squeeze(texture_feat2, 0)
        # texture_feat3 = torch.squeeze(texture_feat3, 0)
        # hist = torch.cat((hist, texture_feat2), dim=0)
        # # hist = torch.cat((hist, texture_feat1, texture_feat2, texture_feat3), dim=0)
        # # print(hist.shape)
        # return (hist, times, filename)

        # # gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)  # 变成黑白
        # x = 500
        # y = 500
        # w = 1000
        # h = 1500
        # img = img[x:x + w, y:y + h]
        # # img = cv2.equalizeHist(img)
        # # clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(10, 5))
        # # gray = clahe.apply(gray)
        # # size = (224, 224)
        # # size = (50, 50)
        # # img = cv2.resize(img, size)
        # hist_0 = cv2.calcHist([img], [0], None, [256], [0, 256])
        # hist_1 = cv2.calcHist([img], [1], None, [256], [0, 256])
        # hist_2 = cv2.calcHist([img], [2], None, [256], [0, 256])
        # hist_0 = torch.Tensor(hist_0)
        # hist_1 = torch.Tensor(hist_1)
        # hist_2 = torch.Tensor(hist_2)
        # hist = torch.stack((hist_0, hist_1, hist_2), 0)
        # # print(hist.shape)
        # hist = Group_2G_R_B(img)
        # hist = Group_GB(img)
        # hist = Group_RB(img)
        # hist = Group_GR(img)
        # hist2 = Group_lab(img)
        # hist = torch.cat((hist, hist2), dim=0)
        # print(hist.shape)


    def __len__(self):
        return len(self.lines)


def normalization(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        f.readline()
        for line in f:
            filename, time_gt,time_pd= line.strip().split(",")

def process_dataset_RB(input_folder, output_folder):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历数据文件夹下的每个子文件夹
    for folder_name in os.listdir(input_folder):
        folder_path = os.path.join(input_folder, folder_name)
        if not os.path.isdir(folder_path):
            continue  # 忽略非文件夹项目

        output_subfolder = os.path.join(output_folder, folder_name)
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)

        # 遍历每个子文件夹下的图像文件
        for filename in os.listdir(folder_path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(folder_path, filename)
                output_path = os.path.join(output_subfolder, filename)

                # 读取图像并进行RB处理
                src_image = cv2.imread(image_path)
                processed_image = Group_RBimg(src_image)

                # 保存处理后的图像
                cv2.imwrite(output_path, processed_image)

if __name__ == "__main__":
    pass
    # generate_all_txt(root="/home/llj/code/test/data_rb", txt_dir="/home/llj/code/test/")
    # split_train_test("/home/llj/code/test/")
    #
    # generate_all_txt2(root="/home/llj/code/test/data2_rb", txt_dir="/home/llj/code/test/")
    # # split_train_test2("/home/llj/code/test/")
    #
    split_data("/home/llj/code/test/")
    # split_data2("/home/llj/code/test/")

    # with open(os.path.join("/home/llj/code/test/", "train.txt"), "r", encoding="utf-8") as f:
    #     data = f.readlines()
    # random.shuffle(data)
    # with open(os.path.join("/home/llj/code/test/", "train.txt"), "w", encoding="utf-8") as f:
    #     lines = f.writelines(data)

    # transform1 = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Resize([224, 224], antialias=True),
    #     # transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
    #     # transforms.RandomRotation(degrees=(0, 180)),
    # ])
    # train_dataset = BatchDataset("/home/llj/code/test/data", "/home/llj/code/test/", "train", transform=transform1)
    # print(train_dataset[0])

    # image = Image.open("/home/llj/code/test/data/20230610/110917_ch01.jpg").convert('RGB')
    # image = transform1(image)
    # print(image.shape)

    # # 输入数据集文件夹和输出文件夹
    # input_folder = "data2"
    # output_folder = "data2_rb"
    #
    # # 处理数据集
    # process_dataset_RB(input_folder, output_folder)

