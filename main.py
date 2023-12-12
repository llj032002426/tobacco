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
from torch.utils.data import Dataset
from group_feature import Group_2G_R_B,Group_GRGB,Group_RG,Group_GR,Group_lab
from textural_feature import fast_glcm_std,fast_glcm_contrast,fast_glcm_dissimilarity,fast_glcm_homogeneity,fast_glcm_ASM,fast_glcm_ENE,fast_glcm_max,fast_glcm_entropy
from PIL import  Image
import os
import numpy as np
import torchvision.transforms as transforms
import datetime
import time

def generate_all_txt(root, txt_dir, num_samples=None):
    """ 筛选符合要求的数据并生成 all.txt 文件 """
    f = open(os.path.join(txt_dir, "all.txt"), "w", encoding="utf-8")
    all_imgs_path = glob.glob(r'/home/llj/code/test/data/*/*.jpg')#数据文件夹路径
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
        times = (d1 - d2).days * 24 + int(file_name[:2])

        # times = int(times / 6)
        # print(times)
        f.write(f"{var},{times}\n")

def generate_all_txt2(root, txt_dir, num_samples=None):
    """ 筛选符合要求的数据并生成 all.txt 文件 """
    f = open(os.path.join(txt_dir, "all2.txt"), "w", encoding="utf-8")
    all_imgs_path = glob.glob(r'/home/llj/code/test/data2/*/*.jpg')#数据文件夹路径
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
        times = (d1 - d2).days * 24 + int(file_name[:2])

        # times = int(times / 6)
        # print(times)
        f.write(f"{var},{times}\n")

def split_train_test(txt_dir, ratio=0.6, seed=123):
    '''拆分训练集和测试集
    '''
    with open(os.path.join(txt_dir, "all.txt"), "r", encoding="utf-8") as f:
        lines = f.readlines()
    total = len(lines)

    random.seed(seed)
    random.shuffle(lines)
    n_train = int(total * ratio)
    train_lines = lines[:n_train]
    val_lines = lines[n_train:]

    with open(os.path.join(txt_dir, "train.txt"), "w", encoding="utf-8") as f:
        for line in train_lines:
            f.write(line)
    with open(os.path.join(txt_dir, "val.txt"), "w", encoding="utf-8") as f:
        for line in val_lines:
            f.write(line)

import os
import random
def split_data(txt_dir, train_ratio=0.6, val_ratio=0.2, seed=123):
    """将数据集按照比例分为训练集、验证集和测试集三个"""
    with open(os.path.join(txt_dir, "all.txt"), "r", encoding="utf-8") as f:
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

    with open(os.path.join(txt_dir, "train.txt"), "w", encoding="utf-8") as f:
        for line in train_lines:
            f.write(line)
    with open(os.path.join(txt_dir, "val.txt"), "w", encoding="utf-8") as f:
        for line in val_lines:
            f.write(line)
    with open(os.path.join(txt_dir, "test.txt"), "w", encoding="utf-8") as f:
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


def hist_ave_2(src):
    L=np.unique(src)
    cdf=(np.histogram(src.flatten(),L.size)[0]/src.size).cumsum()
    cdf=(cdf*L.max()+0.5)
    return np.interp(src.flatten(),L,cdf).reshape(src.shape)
class BatchDataset(Dataset):

    def __init__(self, root, txt_dir, name, transform):
        # self.root = root
        self.transform = transform
        with open(os.path.join(txt_dir, f"{name}.txt"), "r", encoding="utf-8") as f:
            self.lines = f.readlines()

    def __getitem__(self, idx):
        filename, times= self.lines[idx].strip().split(",")
        # image = Image.open(filename).convert('RGB')
        # image = self.transform(image)
        img = cv2.imread(filename, 1)

        # times = np.float32(int(times)/100.0)
        times = np.float32(int(times)/ 100.0)
        # times = np.float32(float(times)/100.0)
        # return (image, times, filename)

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
        # print(hist.shape)

        hist = Group_2G_R_B(img)
        # hist = Group_GRGB(img)
        # hist = Group_RG(img)
        # hist = Group_GR(img)
        # hist = Group_lab(img)

        img = np.array(Image.open(filename).resize((160, 120)).convert('L'))
        # texture_feat = fast_glcm_std(img)
        # texture_feat = fast_glcm_contrast(img)
        # texture_feat = fast_glcm_dissimilarity(img)
        # texture_feat = fast_glcm_homogeneity(img)
        # texture_feat = fast_glcm_ASM(img)
        # texture_feat = fast_glcm_ENE(img)
        texture_feat = fast_glcm_max(img)
        # texture_feat = fast_glcm_entropy(img)
        texture_feat = torch.Tensor(texture_feat)
        texture_feat = torch.reshape(texture_feat, (3, 256, -1))
        hist = torch.cat((hist, texture_feat), dim=2)

        # img = np.array(Image.open(filename).convert('L'))
        # # texture_feat = fast_glcm_std(img)
        # # texture_feat = fast_glcm_contrast(img)
        # # texture_feat = fast_glcm_dissimilarity(img)
        # # texture_feat = fast_glcm_homogeneity(img)
        # texture_feat = fast_glcm_ASM(img)
        # # texture_feat = fast_glcm_ENE(img)
        # # texture_feat = fast_glcm_max(img)
        # # texture_feat = fast_glcm_entropy(img)
        # texture_feat = torch.Tensor(texture_feat)
        # texture_feat = torch.unsqueeze(texture_feat,0)
        # texture_feat = torch.unsqueeze(texture_feat, 0)
        # texture_feat = torch.nn.functional.interpolate(texture_feat, size=(256, 1), mode='bilinear',
        #                                                           align_corners=False)
        # texture_feat = torch.squeeze(texture_feat, 0)
        # hist = torch.cat((hist, texture_feat), dim=0)
        # print(hist.shape)
        return (hist, times, filename)


    def __len__(self):
        return len(self.lines)

if __name__ == "__main__":
    pass
    # generate_all_txt(root="/home/llj/code/test/data", txt_dir="/home/llj/code/test/")
    # split_train_test("/home/llj/code/test/")
    #
    generate_all_txt2(root="/home/llj/code/test/data2", txt_dir="/home/llj/code/test/")
    # split_train_test2("/home/llj/code/test/")

    # split_data("/home/llj/code/test/")
    split_data2("/home/llj/code/test/")

    # with open(os.path.join("/home/llj/code/test/", "1.txt"), "r", encoding="utf-8") as f:
    #     data = f.readlines()
    # random.shuffle(data)
    # with open(os.path.join("/home/llj/code/test/", "1.txt"), "w", encoding="utf-8") as f:
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

