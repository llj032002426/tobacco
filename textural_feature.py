# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import data
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from group_feature import Group_RBimg

def fast_glcm(img, vmin=0, vmax=255, nbit=8, kernel_size=5):
    mi, ma = vmin, vmax
    ks = kernel_size
    h,w = img.shape

    # digitize
    bins = np.linspace(mi, ma+1, nbit+1)
    gl1 = np.digitize(img, bins) - 1
    gl2 = np.append(gl1[:,1:], gl1[:,-1:], axis=1)

    # make glcm
    glcm = np.zeros((nbit, nbit, h, w), dtype=np.uint8)
    for i in range(nbit):
        for j in range(nbit):
            mask = ((gl1==i) & (gl2==j))
            glcm[i,j, mask] = 1

    kernel = np.ones((ks, ks), dtype=np.uint8)
    for i in range(nbit):
        for j in range(nbit):
            glcm[i,j] = cv2.filter2D(glcm[i,j], -1, kernel)

    return glcm


def fast_glcm_mean(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm mean
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    mean = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            mean += glcm[i,j] * i / (nbit)**2
    scaler = MinMaxScaler()
    scaled_mean = scaler.fit_transform(mean)
    return scaled_mean


def fast_glcm_std(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm std
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    mean = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            mean += glcm[i,j] * i / (nbit)**2

    std2 = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            std2 += (glcm[i,j] * i - mean)**2

    std = np.sqrt(std2)
    scaler = MinMaxScaler()
    scaled_std = scaler.fit_transform(std)
    return scaled_std


def fast_glcm_contrast(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm contrast
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    cont = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            cont += glcm[i,j] * (i-j)**2
    scaler = MinMaxScaler()
    scaled_cont = scaler.fit_transform(cont)
    return scaled_cont


def fast_glcm_dissimilarity(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm dissimilarity
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    diss = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            diss += glcm[i,j] * np.abs(i-j)
    scaler = MinMaxScaler()
    scaled_diss = scaler.fit_transform(diss)
    return scaled_diss


def fast_glcm_homogeneity(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm homogeneity
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    homo = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            homo += glcm[i,j] / (1.+(i-j)**2)
    scaler = MinMaxScaler()
    scaled_homo = scaler.fit_transform(homo)
    return scaled_homo


def fast_glcm_ASM(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm asm, energy
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    asm = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            asm  += glcm[i,j]**2

    scaler = MinMaxScaler()
    scaled_asm = scaler.fit_transform(asm)
    return scaled_asm

def ASM(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm asm, energy
    '''
    h,w = img.shape
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    asm = np.zeros((h,w), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            asm  += glcm[i,j]**2
    return asm

def fast_glcm_ENE(img, vmin=0, vmax=255, nbit=8, ks=5):
    asm = ASM(img)
    ene = np.sqrt(asm)
    scaler = MinMaxScaler()
    scaled_ene = scaler.fit_transform(ene)
    return scaled_ene

def fast_glcm_max(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm max
    '''
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    max_  = np.max(glcm, axis=(0,1))
    scaler = MinMaxScaler()
    scaled_ma = scaler.fit_transform(max_)
    return scaled_ma

def fast_glcm_entropy(img, vmin=0, vmax=255, nbit=8, ks=5):
    '''
    calc glcm entropy
    '''
    glcm = fast_glcm(img, vmin, vmax, nbit, ks)
    pnorm = glcm / np.sum(glcm, axis=(0,1)) + 1./ks**2
    ent  = np.sum(-pnorm * np.log(pnorm), axis=(0,1))
    scaler = MinMaxScaler()
    scaled_ent = scaler.fit_transform(ent)
    return scaled_ent

def all_glcm(img):
    mean = fast_glcm_mean(img)
    std = fast_glcm_std(img)
    cont = fast_glcm_contrast(img)
    diss = fast_glcm_dissimilarity(img)
    homo = fast_glcm_homogeneity(img)
    asm = fast_glcm_ASM(img)
    ene = fast_glcm_ENE(img)
    ma = fast_glcm_max(img)
    ent = fast_glcm_entropy(img)

    # 归一化0-1
    scaler = MinMaxScaler()
    scaled_mean = scaler.fit_transform(mean)
    scaled_std = scaler.fit_transform(std)
    scaled_cont = scaler.fit_transform(cont)
    scaled_diss = scaler.fit_transform(diss)
    scaled_homo = scaler.fit_transform(homo)
    scaled_asm = scaler.fit_transform(asm)
    scaled_ene = scaler.fit_transform(ene)
    scaled_ma = scaler.fit_transform(ma)
    scaled_ent = scaler.fit_transform(ent)

    mean_arr = np.mean([scaled_mean, scaled_std, scaled_cont, scaled_diss,scaled_homo,scaled_asm,scaled_ene,scaled_ma,scaled_ent], axis=0)
    return mean_arr

if __name__ == '__main__':
    # pass
    # nbit = 8
    # ks = 5
    # mi, ma = 0, 255
    #
    # img = data.camera()
    # h,w = img.shape
    #
    # img[:,:w//2] = img[:,:w//2]//2+127
    # glcm_mean = fast_glcm_mean(img, mi, ma, nbit, ks)


    # img = '/home/llj/code/test/data2/20230701/082405_ch01.jpg'
    # img = '/home/llj/code/test/data/20230610/184915_ch01.jpg'
    # img = '/home/llj/code/test/data/20230612/010101_ch01.jpg'
    # img = '/home/llj/code/test/data2/20230701/061802_ch01.jpg'
    # img = '/home/llj/code/test/data/20230616/123338_ch01.jpg'
    src = cv2.imread('/home/llj/code/test/data/20230615/122456_ch01.jpg')
    image_array = Group_RBimg(src)
    img = np.array(Image.fromarray(image_array).resize((160,120)).convert('L'))
    # print(type(img))
    # img = np.array(Image.open(img).convert('L'))
    # img = np.array(Image.open(img).resize((160,120)).convert('L'))
    # img = data.camera()
    # h, w = img.shape
    # print(img.shape)

    mean = fast_glcm_mean(img)
    # print(mean)
    # scaler = MinMaxScaler()
    # scaled_data = scaler.fit_transform(mean)
    # print(scaled_data)
    # print(mean.shape)
    # print(type(mean))
    std = fast_glcm_std(img)
    cont = fast_glcm_contrast(img)
    diss = fast_glcm_dissimilarity(img)
    homo = fast_glcm_homogeneity(img)
    asm = fast_glcm_ASM(img)
    ene = fast_glcm_ENE(img)
    ma = fast_glcm_max(img)
    ent = fast_glcm_entropy(img)

    # 归一化0-1
    scaler = MinMaxScaler()
    scaled_mean = scaler.fit_transform(mean)
    scaled_std = scaler.fit_transform(std)
    scaled_cont = scaler.fit_transform(cont)
    scaled_diss = scaler.fit_transform(diss)
    scaled_homo = scaler.fit_transform(homo)
    scaled_asm = scaler.fit_transform(asm)
    scaled_ene = scaler.fit_transform(ene)
    scaled_ma = scaler.fit_transform(ma)
    scaled_ent = scaler.fit_transform(ent)


    # plt.figure(figsize=(10, 4.5))
    # fs = 15
    # plt.subplot(2, 5, 1)
    # plt.tick_params(labelbottom=False, labelleft=False)
    # plt.imshow(img)
    # plt.title('original', fontsize=fs)
    #
    # plt.subplot(2, 5, 2)
    # plt.tick_params(labelbottom=False, labelleft=False)
    # plt.imshow(mean)
    # plt.title('mean', fontsize=fs)
    #
    # plt.subplot(2, 5, 3)
    # plt.tick_params(labelbottom=False, labelleft=False)
    # plt.imshow(std)
    # plt.title('std', fontsize=fs)
    #
    # plt.subplot(2, 5, 4)
    # plt.tick_params(labelbottom=False, labelleft=False)
    # plt.imshow(cont)
    # plt.title('contrast', fontsize=fs)
    #
    # plt.subplot(2, 5, 5)
    # plt.tick_params(labelbottom=False, labelleft=False)
    # plt.imshow(diss)
    # plt.title('dissimilarity', fontsize=fs)
    #
    # plt.subplot(2, 5, 6)
    # plt.tick_params(labelbottom=False, labelleft=False)
    # plt.imshow(homo)
    # plt.title('homogeneity', fontsize=fs)
    #
    # plt.subplot(2, 5, 7)
    # plt.tick_params(labelbottom=False, labelleft=False)
    # plt.imshow(asm)
    # plt.title('ASM', fontsize=fs)
    #
    # plt.subplot(2, 5, 8)
    # plt.tick_params(labelbottom=False, labelleft=False)
    # plt.imshow(ene)
    # plt.title('energy', fontsize=fs)
    #
    # plt.subplot(2, 5, 9)
    # plt.tick_params(labelbottom=False, labelleft=False)
    # plt.imshow(ma)
    # plt.title('max', fontsize=fs)
    #
    # plt.subplot(2, 5, 10)
    # plt.tick_params(labelbottom=False, labelleft=False)
    # plt.imshow(ent)
    # plt.title('entropy', fontsize=fs)
    #
    # plt.tight_layout(pad=0.5)
    # plt.savefig('/home/llj/code/test/output.jpg')
    # plt.show()

    # 归一化图像
    plt.figure(figsize=(10, 4.5))
    fs = 15
    plt.subplot(2, 5, 1)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.imshow(img)
    plt.title('original', fontsize=fs)

    plt.subplot(2, 5, 2)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.imshow(scaled_mean)
    plt.title('mean', fontsize=fs)

    plt.subplot(2, 5, 3)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.imshow(scaled_std)
    plt.title('std', fontsize=fs)

    plt.subplot(2, 5, 4)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.imshow(scaled_cont)
    plt.title('contrast', fontsize=fs)

    plt.subplot(2, 5, 5)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.imshow(scaled_diss)
    plt.title('dissimilarity', fontsize=fs)

    plt.subplot(2, 5, 6)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.imshow(scaled_homo)
    plt.title('homogeneity', fontsize=fs)

    plt.subplot(2, 5, 7)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.imshow(scaled_asm)
    plt.title('ASM', fontsize=fs)

    plt.subplot(2, 5, 8)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.imshow(scaled_ene)
    plt.title('energy', fontsize=fs)

    plt.subplot(2, 5, 9)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.imshow(scaled_ma)
    plt.title('max', fontsize=fs)

    plt.subplot(2, 5, 10)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.imshow(scaled_ent)
    plt.title('entropy', fontsize=fs)

    plt.tight_layout(pad=0.5)
    plt.savefig('/home/llj/code/test/output.jpg')
    plt.show()