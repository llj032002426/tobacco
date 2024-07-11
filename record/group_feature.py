import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms

#2G-R-B
def GroupShow_2G_R_B(src):
    # 使用2g-r-b分离土壤与背景
    src = cv2.resize(src, None, fx=0.2, fy=0.2)
    cv2.imshow('src', src)

    # 转换为浮点数进行计算
    fsrc = np.array(src, dtype=np.float32) / 255.0
    (b, g, r) = cv2.split(fsrc)
    gray = 2 * g - b - r

    # 求取最大值和最小值
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)

    # 计算直方图
    hist = cv2.calcHist([gray], [0], None, [256], [minVal, maxVal])
    plt.plot(hist)
    plt.show()

    # 转换为u8类型，进行otsu二值化
    gray_u8 = np.array((gray - minVal) / (maxVal - minVal) * 255, dtype=np.uint8)
    (thresh, bin_img) = cv2.threshold(gray_u8, -1.0, 255, cv2.THRESH_OTSU)
    cv2.imshow('bin_img', bin_img)

    # 得到彩色的图像
    (b8, g8, r8) = cv2.split(src)
    color_img = cv2.merge([b8 & bin_img, g8 & bin_img, r8 & bin_img])
    cv2.imshow('color_img', color_img)
    (b, g, r) = cv2.split(color_img)
    merged = cv2.merge([b, 2*g-b-r, r])
    cv2.imshow('color_img', merged)

    cv2.waitKey()
    cv2.destroyAllWindows()

def Group_2G_R_B(src):
    # 使用2g-r-b分离土壤与背景
    src = cv2.resize(src, None, fx=0.3, fy=0.3)
    # cv2.imshow('src', src)

    # 转换为浮点数进行计算
    fsrc = np.array(src, dtype=np.float32) / 255.0
    (b, g, r) = cv2.split(fsrc)
    gray = 2 * g - b - r

    # 求取最大值和最小值
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)

    # 计算直方图
    # hist = cv2.calcHist([gray], [0], None, [256], [minVal, maxVal])
    # plt.plot(hist)
    # plt.show()

    # 转换为u8类型，进行otsu二值化
    gray_u8 = np.array((gray - minVal) / (maxVal - minVal) * 255, dtype=np.uint8)
    (thresh, bin_img) = cv2.threshold(gray_u8, -1.0, 255, cv2.THRESH_OTSU)
    # cv2.imshow('bin_img', bin_img)

    # 得到彩色的图像
    (b8, g8, r8) = cv2.split(src)
    color_img = cv2.merge([b8 & bin_img, g8 & bin_img, r8 & bin_img])
    # cv2.imshow('color_img', color_img)

    hist_0 = cv2.calcHist([color_img], [0], None, [256], [0, 256])
    hist_1 = cv2.calcHist([color_img], [1], None, [256], [0, 256])
    hist_2 = cv2.calcHist([color_img], [2], None, [256], [0, 256])
    hist_0 = torch.Tensor(hist_0)
    hist_1 = torch.Tensor(hist_1)
    hist_2 = torch.Tensor(hist_2)
    hist = torch.stack((hist_0, hist_1, hist_2), 0)
    # print(hist.shape)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return hist

#G/(R+G+B)
def GroupShow_GB(src):
    # 使用2g-r-b分离土壤与背景
    src = cv2.resize(src, None, fx=0.3, fy=0.3)
    cv2.imshow('src', src)

    # 转换为浮点数进行计算
    fsrc = np.array(src, dtype=np.float32) / 255.0
    (b, g, r) = cv2.split(fsrc)
    # np.seterr(divide='ignore', invalid='ignore')  # 消除被除数为0的警告
    # gray = g/(r+g+b)
    # gray[np.isnan(gray)] = 0.0
    # gray[np.isinf(gray)] = 0.0
    gray = g - b

    # 求取最大值和最小值
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)

    # 计算直方图
    hist = cv2.calcHist([gray], [0], None, [256], [minVal, maxVal])
    plt.plot(hist)
    plt.show()

    # 转换为u8类型，进行otsu二值化
    gray_u8 = np.array((gray - minVal) / (maxVal - minVal) * 255, dtype=np.uint8)
    (thresh, bin_img) = cv2.threshold(gray_u8, -1.0, 255, cv2.THRESH_OTSU)
    cv2.imshow('bin_img', bin_img)

    # 得到彩色的图像
    (b8, g8, r8) = cv2.split(src)
    color_img = cv2.merge([b8 & bin_img, g8 & bin_img, r8 & bin_img])
    cv2.imshow('color_img', color_img)

    cv2.waitKey()
    cv2.destroyAllWindows()
def Group_GB(src):
    # 使用2g-r-b分离土壤与背景
    src = cv2.resize(src, None, fx=0.3, fy=0.3)
    # cv2.imshow('src', src)

    # 转换为浮点数进行计算
    fsrc = np.array(src, dtype=np.float32) / 255.0
    (b, g, r) = cv2.split(fsrc)
    gray = g - b

    # 求取最大值和最小值
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)

    # 计算直方图
    # hist = cv2.calcHist([gray], [0], None, [256], [minVal, maxVal])
    # plt.plot(hist)
    # plt.show()

    # 转换为u8类型，进行otsu二值化
    gray_u8 = np.array((gray - minVal) / (maxVal - minVal) * 255, dtype=np.uint8)
    (thresh, bin_img) = cv2.threshold(gray_u8, -1.0, 255, cv2.THRESH_OTSU)
    # cv2.imshow('bin_img', bin_img)

    # 得到彩色的图像
    (b8, g8, r8) = cv2.split(src)
    color_img = cv2.merge([b8 & bin_img, g8 & bin_img, r8 & bin_img])
    # cv2.imshow('color_img', color_img)

    hist_0 = cv2.calcHist([color_img], [0], None, [256], [0, 256])
    hist_1 = cv2.calcHist([color_img], [1], None, [256], [0, 256])
    hist_2 = cv2.calcHist([color_img], [2], None, [256], [0, 256])
    # plt.plot(hist_0, label='0', color='blue')
    # plt.plot(hist_1, label='1', color='green')
    # plt.plot(hist_2, label='2', color='red')
    # plt.legend(loc='best')
    # plt.xlim([0, 256])
    # plt.show()
    hist_0 = torch.Tensor(hist_0)
    hist_1 = torch.Tensor(hist_1)
    hist_2 = torch.Tensor(hist_2)
    hist = torch.stack((hist_0, hist_1, hist_2), 0)
    # print(hist_0)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return hist

#R/G
def GroupShow_RB(src):
    # 使用r-b分离土壤与背景
    src = cv2.resize(src, None, fx=0.3, fy=0.3)
    cv2.imshow('src', src)

    # 转换为浮点数进行计算
    fsrc = np.array(src, dtype=np.float32) / 255.0
    (b, g, r) = cv2.split(fsrc)
    gray = r - b

    # print(gray)
    # gray = np.divide(r, g, out=np.zeros_like(r, dtype=np.float64), where=g != 0)
    # gray = np.uint8(gray)
    # print(gray)

    # 求取最大值和最小值
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)

    # 计算直方图
    hist = cv2.calcHist([gray], [0], None, [256], [minVal, maxVal])
    plt.plot(hist)
    plt.show()

    # 转换为u8类型，进行otsu二值化
    gray_u8 = np.array((gray - minVal) / (maxVal - minVal) * 255, dtype=np.uint8)
    (thresh, bin_img) = cv2.threshold(gray_u8, -1.0, 255, cv2.THRESH_OTSU)
    cv2.imshow('bin_img', bin_img)

    # 得到彩色的图像
    (b8, g8, r8) = cv2.split(src)
    color_img = cv2.merge([b8 & bin_img, g8 & bin_img, r8 & bin_img])
    cv2.imshow('color_img', color_img)

    cv2.waitKey()
    cv2.destroyAllWindows()
def Group_RBimg(src):
    # src = cv2.resize(src, None, fx=0.2, fy=0.2)
    fsrc = np.array(src, dtype=np.float32) / 255.0
    (b, g, r) = cv2.split(fsrc)
    gray = r - b
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    gray_u8 = np.array((gray - minVal) / (maxVal - minVal) * 255, dtype=np.uint8)
    (thresh, bin_img) = cv2.threshold(gray_u8, -1.0, 255, cv2.THRESH_OTSU)
    (b8, g8, r8) = cv2.split(src)
    color_img = cv2.merge([b8 & bin_img, g8 & bin_img, r8 & bin_img])
    return color_img
def Group_RB(src):
    # with np.seterr(divide='ignore', invalid='ignore'):
    # 使用2g-r-b分离土壤与背景
    # src = cv2.resize(src, None, fx=0.3, fy=0.3)
    # cv2.imshow('src', src)

    # 转换为浮点数进行计算
    fsrc = np.array(src, dtype=np.float32) / 255.0
    (b, g, r) = cv2.split(fsrc)
    gray = r - b  # 获取红色通道减去蓝色通道得到的灰度图像

    # 计算灰度图像中的最小值、最大值及其对应的位置
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    # 转换为u8类型，进行otsu二值化  将灰度图像线性映射到每个像素值（0-255）范围内，转换为无符号8位整型数组。
    gray_u8 = np.array((gray - minVal) / (maxVal - minVal) * 255, dtype=np.uint8)
    (thresh, bin_img) = cv2.threshold(gray_u8, -1.0, 255, cv2.THRESH_OTSU)
    # cv2.imshow('bin_img', bin_img)

    # 得到彩色的图像
    (b8, g8, r8) = cv2.split(src)
    color_img = cv2.merge([b8 & bin_img, g8 & bin_img, r8 & bin_img])
    # cv2.imshow('color_img', color_img)

    hist_0 = cv2.calcHist([color_img], [0], None, [256], [0, 256])
    hist_1 = cv2.calcHist([color_img], [1], None, [256], [0, 256])
    hist_2 = cv2.calcHist([color_img], [2], None, [256], [0, 256])
    hist_0 = torch.Tensor(hist_0)
    hist_1 = torch.Tensor(hist_1)
    hist_2 = torch.Tensor(hist_2)
    hist = torch.stack((hist_0, hist_1, hist_2), 0)
    # print(hist.shape)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return hist

#G-R
def GroupShow_GR(src):
    # 使用2g-r-b分离土壤与背景
    src = cv2.resize(src, None, fx=0.2, fy=0.2)
    cv2.imshow('src', src)

    # 转换为浮点数进行计算
    fsrc = np.array(src, dtype=np.float32) / 255.0
    (b, g, r) = cv2.split(fsrc)
    gray = g - r

    # 求取最大值和最小值
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)

    # 计算直方图
    hist = cv2.calcHist([gray], [0], None, [256], [minVal, maxVal])
    plt.plot(hist)
    plt.show()

    # 转换为u8类型，进行otsu二值化
    gray_u8 = np.array((gray - minVal) / (maxVal - minVal) * 255, dtype=np.uint8)
    (thresh, bin_img) = cv2.threshold(gray_u8, -1.0, 255, cv2.THRESH_OTSU)
    cv2.imshow('bin_img', bin_img)

    # 得到彩色的图像
    (b8, g8, r8) = cv2.split(src)
    color_img = cv2.merge([b8 & bin_img, g8 & bin_img, r8 & bin_img])
    cv2.imshow('color_img', color_img)

    cv2.waitKey()
    cv2.destroyAllWindows()
def Group_GR(src):
    # with np.seterr(divide='ignore', invalid='ignore'):
    # 使用2g-r-b分离土壤与背景
    src = cv2.resize(src, None, fx=0.3, fy=0.3)
    # cv2.imshow('src', src)

    # 转换为浮点数进行计算
    fsrc = np.array(src, dtype=np.float32) / 255.0
    (b, g, r) = cv2.split(fsrc)
    gray = g-r

    # 求取最大值和最小值
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)

    # 计算直方图
    # hist = cv2.calcHist([gray], [0], None, [256], [minVal, maxVal])
    # plt.plot(hist)
    # plt.show()

    # 转换为u8类型，进行otsu二值化
    gray_u8 = np.array((gray - minVal) / (maxVal - minVal) * 255, dtype=np.uint8)
    (thresh, bin_img) = cv2.threshold(gray_u8, -1.0, 255, cv2.THRESH_OTSU)
    # cv2.imshow('bin_img', bin_img)

    # 得到彩色的图像
    (b8, g8, r8) = cv2.split(src)
    color_img = cv2.merge([b8 & bin_img, g8 & bin_img, r8 & bin_img])
    # cv2.imshow('color_img', color_img)

    hist_0 = cv2.calcHist([color_img], [0], None, [256], [0, 256])
    hist_1 = cv2.calcHist([color_img], [1], None, [256], [0, 256])
    hist_2 = cv2.calcHist([color_img], [2], None, [256], [0, 256])
    # print(hist_0)
    # print(hist_0.shape)
    # print(type(hist_0))
    hist_0 = torch.Tensor(hist_0)
    hist_1 = torch.Tensor(hist_1)
    hist_2 = torch.Tensor(hist_2)
    hist = torch.stack((hist_0, hist_1, hist_2), 0)
    # print(hist.shape)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return hist


def GroupShow_lab():
    # 读取图片
    pic_file = '/home/llj/code/test/data2/20230701/082405_ch01.jpg'
    img_bgr = cv2.imread(pic_file, cv2.IMREAD_COLOR)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    cv2.namedWindow("input", cv2.WINDOW_GUI_NORMAL)
    cv2.imshow("input", img_lab)

    # 分别获取三个通道的ndarray数据
    img_ls = img_lab[:, :, 0]
    img_as = img_lab[:, :, 1]
    img_bs = img_lab[:, :, 2]

    '''按L、A、B三个通道分别计算颜色直方图'''
    ls_hist = cv2.calcHist([img_lab], [0], None, [256], [0, 255])
    as_hist = cv2.calcHist([img_lab], [1], None, [256], [0, 255])
    bs_hist = cv2.calcHist([img_lab], [2], None, [256], [0, 255])
    m,dev = cv2.meanStdDev(img_lab)  #计算L、A、B三通道的均值和方差
    print(m)

    '''显示三个通道的颜色直方图'''
    plt.plot(ls_hist, label='l', color='blue')
    plt.plot(as_hist, label='a', color='green')
    plt.plot(bs_hist, label='b', color='red')
    plt.legend(loc='best')
    plt.xlim([0, 256])
    plt.show()
    cv2.waitKey(0)

def Group_lab(img_bgr):
    # 读取图片
    # img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    src = cv2.resize(img_bgr, None, fx=0.3, fy=0.3)
    fsrc = np.array(src, dtype=np.float32) / 255.0
    (b, g, r) = cv2.split(fsrc)
    gray = r - b
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    gray_u8 = np.array((gray - minVal) / (maxVal - minVal) * 255, dtype=np.uint8)
    (thresh, bin_img) = cv2.threshold(gray_u8, -1.0, 255, cv2.THRESH_OTSU)
    (b8, g8, r8) = cv2.split(src)
    color_img = cv2.merge([b8 & bin_img, g8 & bin_img, r8 & bin_img])
    img_lab = cv2.cvtColor(color_img, cv2.COLOR_BGR2LAB)

    # 分别获取三个通道的ndarray数据
    img_ls = img_lab[:, :, 0]
    img_as = img_lab[:, :, 1]
    img_bs = img_lab[:, :, 2]

    '''按L、A、B三个通道分别计算颜色直方图'''
    ls_hist = cv2.calcHist([img_lab], [0], None, [256], [0, 255])
    as_hist = cv2.calcHist([img_lab], [1], None, [256], [0, 255])
    bs_hist = cv2.calcHist([img_lab], [2], None, [256], [0, 255])
    m,dev = cv2.meanStdDev(img_lab)  #计算L、A、B三通道的均值和方差
    # print(m)

    ls_hist = torch.Tensor(ls_hist)
    as_hist = torch.Tensor(as_hist)
    bs_hist = torch.Tensor(bs_hist)
    hist = torch.stack((ls_hist, as_hist, bs_hist), 0)
    # print(hist)
    # print(hist.shape)
    return hist

if __name__ == "__main__":
    # pass
    # src = cv2.imread('/home/llj/code/test/data2/20230701/061802_ch01.jpg')
    # src = cv2.imread('/home/llj/code/test/data/20230616/123338_ch01.jpg')
    src = cv2.imread('/home/llj/code/test/data/20230615/122456_ch01.jpg')
    # src = cv2.imread('/home/llj/code/test/data/20230610/184915_ch01.jpg')
    # src = cv2.imread('/home/llj/code/test/data/20230612/010101_ch01.jpg')
    # GroupShow_2G_R_B(src)
    # Group_2G_R_B(src)

    # GroupShow_GB(src)
    # Group_GB(src)

    # GroupShow_RB(src)
    # Group_RB(src)

    # GroupShow_GR(src)
    # Group_GR(src)

    # GroupShow_lab()
    # Group_lab(src)

    image_array = Group_RBimg(src)
    output_directory = "output.jpg"
    cv2.imwrite(output_directory, image_array)
    # # cv2.imshow('color_img', image_array)
    # # cv2.waitKey()
    # # cv2.destroyAllWindows()
    # hist_0 = cv2.calcHist([image_array], [0], None, [256], [0, 256])
    # hist_1 = cv2.calcHist([image_array], [1], None, [256], [0, 256])
    # hist_2 = cv2.calcHist([image_array], [2], None, [256], [0, 256])
    # # plt.plot(hist_0, label='r', color='red')
    # # plt.plot(hist_1, label='g', color='green')
    # plt.plot(hist_2, label='b', color='blue')
    # plt.show()

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Resize([170, 120], antialias=True),
    #     transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
    #     transforms.RandomRotation(degrees=(0, 180)),
    # ])
    # image = transform(image_array)
    # print(image.shape)

    # plt.legend(loc='best')
    # plt.xlim([0, 256])
    # plt.show()








