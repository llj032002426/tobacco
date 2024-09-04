# # 将差值大于6的删除
# import re
# # 输入文件和输出文件的路径
# input_file = "./results/temp_ps64_n6.csv"
# output_file = "1.txt"
# # input_file = "./results/101_avg.csv"
# # output_file = "./results/101_FFT<=6.txt"
# # 正则表达式模式
# # pattern = r'/.*?(\d+),(\d+)\s*$'
# pattern = r'/.*?(\d+\.\d+),(\d+\.\d+)\s*$'
# # 保存匹配行的列表
# matched_lines = []
# # 打开输入文件
# with open(input_file, 'r') as fin:
#     # 遍历输入文件的每一行
#     for line in fin:
#         # 匹配模式
#         match = re.search(pattern, line)
#         if match:
#             # 提取两个数字
#             # num1 = int(match.group(1))
#             # num2 = int(match.group(2))
#             num1 = float(match.group(1))
#             num2 = float(match.group(2))
#             # 检查差异是否大于4
#             if abs(num1 - num2) <= 2:
#                 # 将匹配的行添加到列表中
#                 matched_lines.append(line)
# # 根据每行的第一个数字进行排序
# # matched_lines.sort(key=lambda x: int(re.search(pattern, x).group(1)))
# # matched_lines.sort(key=lambda x: float(re.search(pattern, x).group(1)))
# # 将排序后的结果写入到输出文件
# with open(output_file, 'w') as fout:
#     fout.writelines(matched_lines)
# print("筛选并排序完成，结果已写入到", output_file)


#改数值出现次数
from datetime import datetime, timedelta
def parse_time_from_address(address):
    # 从地址中提取时间信息
    time_str1 = address.split('/')[-1].split('_')[0]
    time_str2 = address.split('/')[-2].split('_')[0]
    time_str = time_str2 + time_str1
    # print(time_str,time_str1,time_str2)
    return datetime.strptime(time_str, "%Y%m%d%H%M%S")
def within_5_minutes(time1, time2):
    # 判断两个时间是否在5分钟内
    return abs((time2 - time1).total_seconds()) <= 5 * 60
def process_file(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    # 解析地址中的时间信息并排序
    lines.sort(key=lambda x: parse_time_from_address(x.split(',')[0]))
    output_lines = []
    prev_time = None
    count_within_5_minutes = 0
    for line in lines:
        address, value1, value2 = line.strip().split(',')
        time = parse_time_from_address(address)
        if prev_time is None or not within_5_minutes(prev_time, time):
            # 如果与前一行的时间间隔超过5分钟，重置计数器
            count_within_5_minutes = 0
            prev_time = time
        else:
            # 如果与前一行的时间间隔在5分钟内，增加计数器
            count_within_5_minutes += 1
        # 只保留每5分钟内的前两行
        if count_within_5_minutes <= 3:
            output_lines.append(line)
    # 写入输出文件
    with open(output_file, 'w') as f:
        f.writelines(output_lines)
# process_file("./results/ps64_temp2_avg.csv", "1.txt")


#不同参数MAE对比图
# import matplotlib.pyplot as plt
# from matplotlib import rcParams
# rcParams.update({'font.size': 15})
# # patchsize
# x=[64,80,96,112]
# # x=[48,64,80,96]
# y=[0.63,0.53,0.49,0.69]
# # y=[1.41,0.84,1.29,1.15]
# plt.plot(x, y)
# plt.xticks([64,80,96,112])
# # plt.xticks([48,64,80,96])
# plt.yticks([0.25,0.5,0.75,1])
# # plt.yticks([0.75,1,1.25,1.5])
# # x=[48,64,80,96,112]
# # y=[4.98,3.46,3.47,2.77,4.18]
# # plt.plot(x, y)
# # plt.xticks([48,64,80,96,112])
# # plt.yticks([2,3,4,5])
# plt.xlabel('patchsize')
# plt.ylabel('MAE')
# # plt.title('Temperature MAE In Different Patchsize')
# plt.show()
#
# nblock
# x=[2,4,6,8]
# y=[0.89,1.17,0.84,1.04]
# plt.plot(x, y)
# plt.xticks([2,4,6,8])
# plt.yticks([0.5,0.75,1,1.25])
# plt.xlabel('Kblock')
# plt.ylabel('MAE')
# # plt.title('Temperature MAE In Different Kblock')
# plt.show()
# # nblock
# x=[2,4,6,8]
# y=[0.87,0.59,0.49,0.60]
# plt.plot(x, y)
# plt.xticks([2,4,6,8])
# plt.yticks([0.25,0.5,0.75,1])
# plt.xlabel('nblock')
# plt.ylabel('MAE')
# # plt.title('Humidity MAE In Different Nblock')
# plt.show()



# #根据输出文件计算准确率
# count = 0
# total = 0
# loss = 0
# with open("./results/temp_ps64_n6.csv", "r", encoding="utf-8-sig") as f:
# # with open("./results/ps96_houyi1h_data2.csv", "r", encoding="utf-8-sig") as f:
# # with open("./results/ResNet18_hum2_avg.csv", "r", encoding="utf-8-sig") as f:
# # with open("./results/patchsize96_2_FFTall<=9.txt", "r", encoding="utf-8-sig") as f:
# # with open("./results/101_avg.csv", "r", encoding="utf-8-sig") as f:
# # with open("./results/101_FFT<=6.txt", "r", encoding="utf-8-sig") as f:
# # with open("./results/Resnet50.csv", "r", encoding="utf-8-sig") as f:
#     f.readline()
#     for line in f:
#         total += 1
#         filename, time_gt, time_pd = line.strip().split(",")
#         # loss += abs(int(time_gt) - int(time_pd))
#         # if abs(int(time_gt) - int(time_pd)) <= 6:
#         loss += abs(float(time_gt) - float(time_pd))
#         if abs(float(time_gt) - float(time_pd)) <= 2:
#             count += 1
# print("loss:{}, acc:{:.6f}".format(loss/total, count/total))


# def keep_common_prefix_lines(file1, file2, output_file):
#     with open(file1, 'r') as f1, open(file2, 'r') as f2:
#         lines1 = f1.readlines()
#         lines2 = f2.readlines()
#
#     common_lines = []
#     for line1 in lines1:
#         prefix1 = line1.strip().split(",")[0]  # 获取第一个单词作为前缀
#
#         for line2 in lines2:
#             prefix2 = line2.strip().split(",")[0]
#
#             if prefix1 == prefix2:
#                 common_lines.append(line1.strip() + '\n')
#                 break
#
#     with open(output_file, 'w') as f_out:
#         f_out.writelines(common_lines)
#
# # 请替换为你的txt文件路径
# file1_path = 'all2_rb.txt'
# file2_path = './results/best.txt'
# output_file = 'output.txt'
#
# keep_common_prefix_lines(file1_path, file2_path, output_file)
# print("筛选并排序完成，结果已写入到", output_file)


#去掉loss中的无用，只用数值用于绘图
from matplotlib import rcParams
rcParams.update({'font.size': 15})
def process_txt_file(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            # 分割每行内容
            parts = line.strip().split()
            # 提取epoch、trainloss和valloss
            epoch = parts[4].strip()
            epoch = epoch[7:]
            train_loss = parts[8].strip()
            train_loss = train_loss[5:]
            val_loss = parts[9].strip()
            val_loss = val_loss[9:]
            # 写入到输出文件
            f_out.write(f"{epoch},{train_loss},{val_loss}\n")

# 请替换为你的输入文件路径和输出文件路径
input_file_path = '1.txt'
output_file_path = 'loss2.txt'
process_txt_file(input_file_path, output_file_path)

#绘图
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
def plot_loss_from_file(loss_file):
    epochs = []
    train_losses = []
    val_losses = []

    with open(loss_file, 'r') as file:
        lines = file.readlines()[:201]  # 从第二行到第200行（索引从0开始）
        for line in lines:
            epoch, train_loss, val_loss = map(float, line.strip().split(","))
            epochs.append(epoch)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

    plt.plot(epochs[0:], train_losses[0:], label='训练损失')
    plt.plot(epochs[0:], val_losses[0:], label='验证损失')
    plt.xlabel('轮次')
    plt.ylabel('损失', rotation=0, verticalalignment='top', horizontalalignment='right')
    # plt.title('Humidity Training and Validation Losses')
    plt.legend()
    # plt.grid(True)
    # plt.show()
    plt.savefig('./emf_pic/svg/hum_loss.svg', format='svg', dpi=150)
# 请替换为你的训练结果损失文件路径
loss_file_path = 'loss.txt'
# plot_loss_from_file(loss_file_path)


#将测试结果变成两个数字的绝对值差
def process_txt_file(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            # 分割每行内容
            parts = line.strip().split(',')
            # 提取数字1和数字2
            # num1 = int(parts[1])
            # num2 = int(parts[2])
            num1 = float(parts[1])
            num2 = float(parts[2])
            # 计算两个数字的差的绝对值
            result = str(abs(num1 - num2))
            # 将结果写入输出文件
            f_out.write("{}\n".format(
                result
            ))
# 请替换为你的输入文件路径和输出文件路径
input_file_path = './results/hum_ps96_n6_noFFT_2.csv'
output_file_path = './new_results/hum/model/Global-Local Transformer_pd.txt'
# process_txt_file(input_file_path, output_file_path)

#对已知结果的差 绘制CS曲线对比
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.sans-serif']=['SimHei']  # 中文显示
plt.rcParams['svg.fonttype'] = 'none' #保存为svg时文字才不会变成线条
rcParams.update({'font.size': 15})
def calculate_cs_scores(input_files):
    cs_scores_data = {}
    for input_file in input_files:
        numbers = []
        # 读取文件并将数字添加到列表中
        with open(input_file, 'r') as f:
            for line in f:
                # number = int(line.strip())
                number = float(line.strip())
                numbers.append(number)
        # 对数字列表进行排序
        sorted_numbers = sorted(numbers)
        total_samples = len(sorted_numbers)
        # 计算不超过每个阈值的数字的个数，并计算CS(α)
        cs_scores = []
        # thresholds = range(4)  # α从0到6
        thresholds = [i * 0.25 for i in range(5)] #hum
        # thresholds = [i * 0.5 for i in range(7)] #temp
        for threshold in thresholds:
            count_below_threshold = sum(1 for num in sorted_numbers if num <= threshold)
            cs_score = count_below_threshold / total_samples * 100
            cs_scores.append(cs_score)
        cs_scores_data[input_file] = cs_scores
    return thresholds, cs_scores_data
def plot_cs_curves(thresholds, cs_scores_data):
    # colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'black']  # 颜色列表
    markers ='.ov^<>*12348spPhH + xXdD'
    for index, (input_file, cs_scores) in enumerate(cs_scores_data.items()):
        # label = input_file[14:]
        # label = label[:-4]
        # label = input_file[29:]
        # label = input_file[26:]
        # label = input_file[25:]
        # label = input_file[28:]
        # label = input_file[25:]
        label = input_file[24:]
        label = label[:-7]
        # plt.plot(thresholds, cs_scores, linestyle='-') #无对比单个曲线
        plt.plot(thresholds, cs_scores, linestyle='-', label=label, marker=markers[index % len(markers)])
        # plt.plot(thresholds, cs_scores, linestyle='-', label=label, color=colors[index % len(colors)])

    plt.xlabel('阈值')
    plt.ylabel('CS(%)', rotation=0, verticalalignment='top', horizontalalignment='right')
    # plt.title('Humidity Cumulative Score Curve')
    # plt.title('Temperature Cumulative Score Curve')
    # plt.grid(True)
    # plt.xticks(range(3))  # 设置x轴刻度为0到6
    plt.xticks([i * 0.25 for i in range(5)])
    # plt.xticks([i * 0.5 for i in range(7)])
    plt.yticks(range(0, 101, 10))  # 设置y轴刻度为0到100，间隔为10
    # plt.legend()#无对比单个曲线时注释掉
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)#模型对比时使用
    # plt.subplots_adjust(bottom=0.15,top=0.95,left=0.18,right=0.95)
    plt.subplots_adjust(bottom=0.4, top=0.9,left=0.18,right=0.9)#模型对比时使用
    # plt.show()
    plt.savefig('./emf_pic/svg/图8(b).svg', format='svg', dpi=150)
# 请替换为你的输入文件路径列表
# input_files = ['./hum/patchsize/patchsize64_pd.txt', './hum/patchsize/patchsize80_pd.txt', './hum/patchsize/patchsize96_pd.txt', './hum/patchsize/patchsize112_pd.txt']
# input_files = ['./hum/Kblock/Kblock2_pd.txt', './hum/Kblock/Kblock4_pd.txt', './hum/Kblock/Kblock6_pd.txt', './hum/Kblock/Kblock8_pd.txt']
# input_files = ['./hum/model/ours_pd.txt', './hum/model/Global-Local Transformer_pd.txt', './hum/model/SSRNet_pd.txt', './hum/model/ResNet50_pd.txt', './hum/model/ResNet18_pd.txt', './hum/model/VGG16_pd.txt']
# input_files = ['./hum/Kblock/Kblock6_pd.txt'] #无对比单个曲线
# input_files = ['./temp/patchsize/patchsize48_pd.txt', './temp/patchsize/patchsize64_pd.txt', './temp/patchsize/patchsize80_pd.txt', './temp/patchsize/patchsize96_pd.txt']
# input_files = ['./temp/Kblock/Kblock2_pd.txt', './temp/Kblock/Kblock4_pd.txt', './temp/Kblock/Kblock6_pd.txt', './temp/Kblock/Kblock8_pd.txt']
# input_files = ['./temp/model/ours_pd.txt', './temp/model/Global-Local Transformer_pd.txt', './temp/model/SSRNet_pd.txt', './temp/model/ResNet50_pd.txt', './temp/model/ResNet18_pd.txt', './temp/model/VGG16_pd.txt']
# input_files = ['./temp/Kblock/Kblock6_pd.txt'] #无对比单个曲线
# input_files = ['./temp/Kblock/Kblock6_pd.txt', './hum/Kblock/Kblock6_pd.txt']
# input_files = ['./CS/Temperature.txt', './CS/Humidity.txt']

# input_files = ['./new_results/湿度.txt', './new_results/温度.txt']
# input_files = ['./new_results/temp/patchsize/patchsize48_pd.txt', './new_results/temp/patchsize/patchsize64_pd.txt', './new_results/temp/patchsize/patchsize80_pd.txt', './new_results/temp/patchsize/patchsize96_pd.txt']
# input_files = ['./new_results/temp/Kblock/Kblock2_pd.txt', './new_results/temp/Kblock/Kblock4_pd.txt', './new_results/temp/Kblock/Kblock6_pd.txt', './new_results/temp/Kblock/Kblock8_pd.txt']
# input_files = ['./new_results/temp/model/ours_pd.txt', './new_results/temp/model/Global-Local Transformer_pd.txt', './new_results/temp/model/SSRNet_pd.txt', './new_results/temp/model/VGG16_pd.txt', './new_results/temp/model/AlexNet_pd.txt', './new_results/temp/model/Ghostnet_pd.txt', './new_results/temp/model/EfficientNetV2_pd.txt']
# input_files = ['./new_results/hum/patchsize/patchsize64_pd.txt', './new_results/hum/patchsize/patchsize80_pd.txt', './new_results/hum/patchsize/patchsize96_pd.txt', './new_results/hum/patchsize/patchsize112_pd.txt']
# input_files = ['./new_results/hum/Kblock/Kblock2_pd.txt', './new_results/hum/Kblock/Kblock4_pd.txt', './new_results/hum/Kblock/Kblock6_pd.txt', './new_results/hum/Kblock/Kblock8_pd.txt']
input_files = ['./new_results/hum/model/ours_pd.txt', './new_results/hum/model/Global-Local Transformer_pd.txt', './new_results/hum/model/SSRNet_pd.txt', './new_results/hum/model/VGG16_pd.txt', './new_results/hum/model/AlexNet_pd.txt', './new_results/hum/model/Ghostnet_pd.txt', './new_results/hum/model/EfficientNetV2_pd.txt']

# 计算CS(α)并绘制CS曲线对比
# thresholds, cs_scores_data = calculate_cs_scores(input_files)
# plot_cs_curves(thresholds, cs_scores_data)



# import cv2
# import matplotlib.pyplot as plt
# import numpy as np
# img0 = cv2.imread('/home/llj/code/test/data_rb/20230615/122456_ch01.jpg')
# img1 = cv2.resize(img0, dsize=None, fx=0.5, fy=0.5)
# # img1 = cv2.resize(img0, (170, 120), fx=0.5, fy=0.5)
# img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# plt.imshow(img2)
# # plt.show()
# # plt.rcParams['font.family'] = 'Arial Unicode MS'
# f = np.fft.fft2(img2)
# fshift = np.fft.fftshift(f)  # 将0频率分量移动到中心
# magnitude_spectrum = 20 * np.log(np.abs(fshift))
# # print(magnitude_spectrum.shape)
# plt.xticks([])  # 除去刻度线
# plt.yticks([])
# # plt.title("频谱图")
# plt.imshow(magnitude_spectrum, cmap='gray')
# # plt.show()

#展示FFT
# from PIL import  Image
# import torchvision.transforms as transforms
# import torch
# import numpy as np
# filename = '/home/llj/code/test/data_rb/20230615/122456_ch01.jpg'
# image = Image.open(filename).convert('RGB')
# transform1 = transforms.Compose([
#         transforms.ToTensor(),
#         # transforms.Resize([120, 170], antialias=True)
# ])
# image = transform1(image)
# y = transforms.ToPILImage()(image)  # 将torch.Tensor类型图像转换为PIL图像
# # 将PIL图像转换为灰度图像
# y = y.convert("L")
# # 将灰度图像转换为NumPy数组
# y = np.array(y)
# f = np.fft.fft2(y)
# fshift = np.fft.fftshift(f)  # 将0频率分量移动到中心
# magnitude_spectrum = 20 * np.log(np.abs(fshift))
# plt.imshow(magnitude_spectrum, cmap='gray')
# plt.title('Image')
# plt.xticks([])  # 除去刻度线
# plt.yticks([])
# plt.show()

# from PIL import  Image
# import torchvision.transforms as transforms
# import torch
# import numpy as np
# filename = '/home/llj/code/test/data_rb/20230615/122456_ch01.jpg'
# image = Image.open(filename).convert('RGB')
# transform1 = transforms.Compose([
#         transforms.ToTensor(),
#         # transforms.Resize([120, 170], antialias=True)
# ])
# image = transform1(image)
# gray_img = torch.mean(image, dim=0, keepdim=True)
# gray_img = gray_img.squeeze(0)
# # xfreq = torch.fft.fft2(image).real.to(torch.float32)
# xfreq = torch.fft.fft2(gray_img)
# xfreq = torch.fft.fftshift(xfreq)
# # print(xfreq)
# image = torch.abs(xfreq)
# print(image.shape)
# # image_permuted = image.permute(1, 2, 0)
# image_np = (image.numpy() * 255).astype('uint8')
# pil_image = Image.fromarray(image_np)
# plt.imshow(pil_image, cmap='gray')
# plt.title('Image')
# plt.xticks([])  # 除去刻度线
# plt.yticks([])
# plt.show()

# from PIL import  Image
# import torchvision.transforms as transforms
# import torch
# import numpy as np
# filename = '/home/llj/code/test/data_rb/20230615/122456_ch01.jpg'
# image = Image.open(filename).convert('RGB')
# transform1 = transforms.Compose([
#         transforms.ToTensor(),
#         # transforms.Resize([120, 170], antialias=True)
# ])
# image = transform1(image)
# freqinput = torch.sum(image, dim=0, keepdim=True) / 3.0
# input_permuted = freqinput.permute(1, 2, 0)  # 将 channels 移动到最后一维
# input_complex = input_permuted.type(torch.complex64)  # 将输入转换为复数张量
# freq_representation = torch.fft.fftn(input_complex, dim=(0, 1))  # 对 height 和 width 维度进行傅里叶变换
# xfreq = torch.fft.fftshift(freq_representation, dim=(0, 1))
# image = torch.abs(xfreq)
# # print(image.shape)
# # image_permuted = image.permute(1, 2, 0)
# image_np = (image.numpy() * 255).astype('uint8')
# pil_image = Image.fromarray(image_np)
# plt.imshow(pil_image, cmap='gray')
# plt.title('Image')
# plt.xticks([])  # 除去刻度线
# plt.yticks([])
# plt.show()
# # 例如，你可以将实部和虚部合并为一个张量，形状为 (batch_size, height, width, 2)，表示复数的实部和虚部
# xfreq = torch.stack((freq_representation.real, freq_representation.imag), dim=-1)
# xfreq = xfreq.squeeze(-2)
# xfreq = xfreq.permute(0, 3, 1, 2)


# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# import torchvision.transforms as transforms
#
# import os
#
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
# import torch
# import numpy as np
# import cv2
#
# # 读取图像
# # 图像转tensor！
# # 读取图像
# image = Image.open(r'/home/llj/code/test/data_rb/20230615/122456_ch01.jpg')  # 替换为你的图像文件路径
# # 定义转换
# transform = transforms.ToTensor()
# # 将图像转换为张量
# tensor = transform(image)
#
# # 进行傅里叶变换
# fre = torch.fft.fftn(tensor, dim=(-2, -1))  # 在图像的最后两个维度上执行傅里叶变换
# fre_m = torch.abs(fre)  # 幅度谱，求模得到
# fre_p = torch.angle(fre)  # 相位谱，求相角得到
#
# # 把相位设为常数
# constant = torch.mean(fre_m)
# fre_ = constant * torch.exp(1j * fre_p)  # 把幅度谱和相位谱再合并为复数形式的频域图数据
# img_onlyphase = torch.abs(torch.fft.ifftn(fre_, dim=(-2, -1)))  # 还原为空间域图像
#
# # 把振幅设为常数
# constant = torch.mean(fre_p)
# fre_ = fre_m * torch.exp(1j * constant)
# img_onlymagnitude = torch.abs(torch.fft.ifftn(fre_, dim=(-2, -1)))
#
# # tensor转图像！
# # 定义转换
# transform = transforms.ToPILImage()
# # 将张量转换为图像
# image2 = transform(fre_m)
# # 显示图像
# plt.imshow(image2)
# plt.axis('off')  # 不显示坐标轴
# plt.show()


# #将标签文件的内容+1或-1
# input_file = "all2_rb.txt"
# output_file = "all2_houyi1h.txt"
#
# with open(input_file, 'r') as file_in:
#     with open(output_file, 'w') as file_out:
#         for line in file_in:
#             parts = line.split(",")  # 按空格分割每行的内容
#             address = parts[0]  # 地址部分
#             number = int(parts[1]) + 1  # 数字部分加一
#             new_line = f"{address},{number}\n"  # 新的一行内容
#             file_out.write(new_line)  # 写入新文件


# #将时间标签改为温度标签
# array = [0.0] * 150
# for i in range(1, 16):
#     array[i] = 38.0  # 在Python中，索引是从0开始的
# rate_of_increase = (40 - 38) / 4
# for i in range(16, 20):
#     temperature = 38 + (i - 15) * rate_of_increase
#     array[i] = round(temperature, 1)  # 保留一位小数
# for i in range(20, 36):
#     array[i] = 40.0  # 在Python中，索引是从0开始的
# rate_of_increase = (42 - 40) / 6
# for i in range(36, 42):
#     temperature = 40 + (i - 35) * rate_of_increase
#     array[i] = round(temperature, 1)  # 保留一位小数
# for i in range(42, 60):
#     array[i] = 42.0
# rate_of_increase = (46 - 42) / 12
# for i in range(60, 72):
#     temperature = 42 + (i - 59) * rate_of_increase
#     array[i] = round(temperature, 1)  # 保留一位小数
# for i in range(72, 82):
#     array[i] = 46.0
# rate_of_increase = (48 - 46) / 4
# for i in range(82, 86):
#     temperature = 46 + (i - 81) * rate_of_increase
#     array[i] = round(temperature, 1)  # 保留一位小数
# for i in range(86, 98):
#     array[i] = 48.0
# rate_of_increase = (52 - 48) / 4
# for i in range(98, 102):
#     temperature = 48 + (i - 97) * rate_of_increase
#     array[i] = round(temperature, 1)  # 保留一位小数
# for i in range(102, 112):
#     array[i] = 52.0
# rate_of_increase = (54 - 52) / 2
# for i in range(112, 114):
#     temperature = 52 + (i - 111) * rate_of_increase
#     array[i] = round(temperature, 1)  # 保留一位小数
# for i in range(114, 124):
#     array[i] = 54.0
# rate_of_increase = (58 - 54) / 4
# for i in range(124, 128):
#     temperature = 54 + (i - 123) * rate_of_increase
#     array[i] = round(temperature, 1)  # 保留一位小数
# for i in range(128, 138):
#     array[i] = 58.0
# rate_of_increase = (62 - 58) / 6
# for i in range(138, 144):
#     temperature = 58 + (i - 137) * rate_of_increase
#     array[i] = round(temperature, 1)  # 保留一位小数
# array[144] = 62.0
# # 打印结果查看
# # print(array[83])
# input_file = "val_rb1.txt"
# output_file = "val_temp.txt"
# with open(input_file, 'r') as file_in:
#     with open(output_file, 'w') as file_out:
#         for line in file_in:
#             parts = line.split(",")  # 按空格分割每行的内容
#             address = parts[0]  # 地址部分
#             time = int(parts[1])  # 时间标签部分
#             temperature = array[time]  # 转换为温度
#             new_line = f"{address},{temperature}\n"  # 新的一行内容
#             file_out.write(new_line)  # 写入新文件

# #将时间标签改为温度标签data2
# array = [0.0] * 150
# for i in range(1, 11):
#     array[i] = 38.0
# rate_of_increase = (40 - 38) / 6
# for i in range(11, 17):
#     temperature = 38 + (i - 10) * rate_of_increase
#     array[i] = round(temperature, 1)  # 保留一位小数
# for i in range(17, 39):
#     array[i] = 40.0
# rate_of_increase = (42 - 40) / 10
# for i in range(39, 49):
#     temperature = 40 + (i - 38) * rate_of_increase
#     array[i] = round(temperature, 1)  # 保留一位小数
# for i in range(49, 71):
#     array[i] = 42.0
# rate_of_increase = (46 - 42) / 12
# for i in range(71, 83):
#     temperature = 42 + (i - 70) * rate_of_increase
#     array[i] = round(temperature, 1)  # 保留一位小数
# for i in range(83, 93):
#     array[i] = 46.0
# rate_of_increase = (48 - 46) / 4
# for i in range(93, 97):
#     temperature = 46 + (i - 92) * rate_of_increase
#     array[i] = round(temperature, 1)  # 保留一位小数
# for i in range(97, 109):
#     array[i] = 48.0
# rate_of_increase = (52 - 48) / 4
# for i in range(109, 113):
#     temperature = 48 + (i - 108) * rate_of_increase
#     array[i] = round(temperature, 1)  # 保留一位小数
# for i in range(113, 123):
#     array[i] = 52.0
# rate_of_increase = (54 - 52) / 2
# for i in range(123, 125):
#     temperature = 52 + (i - 122) * rate_of_increase
#     array[i] = round(temperature, 1)  # 保留一位小数
# for i in range(125, 135):
#     array[i] = 54.0
# rate_of_increase = (58 - 54) / 4
# for i in range(135, 139):
#     temperature = 54 + (i - 134) * rate_of_increase
#     array[i] = round(temperature, 1)  # 保留一位小数
# for i in range(139, 149):
#     array[i] = 58.0
# # rate_of_increase = (62 - 58) / 6
# # for i in range(138, 144):
# #     temperature = 58 + (i - 137) * rate_of_increase
# #     array[i] = round(temperature, 1)  # 保留一位小数
# # array[144] = 62.0
# # 打印结果查看
# # print(array[83])
# input_file = "all2_houyi1h.txt"
# output_file = "all2_temp2.txt"
#
# with open(input_file, 'r') as file_in:
#     with open(output_file, 'w') as file_out:
#         for line in file_in:
#             parts = line.split(",")  # 按空格分割每行的内容
#             address = parts[0]  # 地址部分
#             time = int(parts[1])  # 时间标签部分
#             temperature = array[time]  # 转换为温度
#             new_line = f"{address},{temperature}\n"  # 新的一行内容
#             file_out.write(new_line)  # 写入新文件

# #将时间标签改为湿度标签
# array = [0.0] * 150
# for i in range(1, 16):
#     array[i] = 38.0  # 在Python中，索引是从0开始的
# rate_of_increase = (39 - 38) / 4
# for i in range(16, 20):
#     temperature = 38 + (i - 15) * rate_of_increase
#     array[i] = round(temperature, 1)  # 保留一位小数
# for i in range(20, 36):
#     array[i] = 39.0  # 在Python中，索引是从0开始的
# rate_of_increase = (37 - 39) / 6
# for i in range(36, 42):
#     temperature = 39 + (i - 35) * rate_of_increase
#     array[i] = round(temperature, 1)  # 保留一位小数
# for i in range(42, 60):
#     array[i] = 37.0
# rate_of_increase = (38 - 37) / 12
# for i in range(60, 72):
#     temperature = 37 + (i - 59) * rate_of_increase
#     array[i] = round(temperature, 1)  # 保留一位小数
# for i in range(72, 82):
#     array[i] = 38.0
# rate_of_increase = (39 - 38) / 4
# for i in range(82, 86):
#     temperature = 38 + (i - 81) * rate_of_increase
#     array[i] = round(temperature, 1)  # 保留一位小数
# for i in range(86, 98):
#     array[i] = 39.0
# rate_of_increase = (39.5 - 39) / 4
# for i in range(98, 102):
#     temperature = 39 + (i - 97) * rate_of_increase
#     array[i] = round(temperature, 1)  # 保留一位小数
# for i in range(102, 112):
#     array[i] = 39.5
# rate_of_increase = (40 - 39.5) / 2
# for i in range(112, 114):
#     temperature = 39.5 + (i - 111) * rate_of_increase
#     array[i] = round(temperature, 1)  # 保留一位小数
# for i in range(114, 124):
#     array[i] = 40.0
# rate_of_increase = (40 - 40) / 4
# for i in range(124, 128):
#     temperature = 40 + (i - 123) * rate_of_increase
#     array[i] = round(temperature, 1)  # 保留一位小数
# for i in range(128, 138):
#     array[i] = 40.0
# rate_of_increase = (41 - 40) / 6
# for i in range(138, 144):
#     temperature = 40 + (i - 137) * rate_of_increase
#     array[i] = round(temperature, 1)  # 保留一位小数
# array[144] = 41.0
# # 打印结果查看
# # print(array[83])
# input_file = "all_rb.txt"
# output_file = "all_hum.txt"
# with open(input_file, 'r') as file_in:
#     with open(output_file, 'w') as file_out:
#         for line in file_in:
#             parts = line.split(",")  # 按空格分割每行的内容
#             address = parts[0]  # 地址部分
#             time = int(parts[1])  # 时间标签部分
#             temperature = array[time]  # 转换为温度
#             new_line = f"{address},{temperature}\n"  # 新的一行内容
#             file_out.write(new_line)  # 写入新文件

# #将时间标签改为湿度标签data2
# array = [0.0] * 150
# for i in range(1, 11):
#     array[i] = 38.0
# rate_of_increase = (39.5 - 38) / 6
# for i in range(11, 17):
#     temperature = 38 + (i - 10) * rate_of_increase
#     array[i] = round(temperature, 1)  # 保留一位小数
# for i in range(17, 39):
#     array[i] = 39.5
# rate_of_increase = (37 - 39.5) / 10
# for i in range(39, 49):
#     temperature = 39.5 + (i - 38) * rate_of_increase
#     array[i] = round(temperature, 1)  # 保留一位小数
# for i in range(49, 71):
#     array[i] = 37.0
# rate_of_increase = (38 - 37) / 12
# for i in range(71, 83):
#     temperature = 37 + (i - 70) * rate_of_increase
#     array[i] = round(temperature, 1)  # 保留一位小数
# for i in range(83, 93):
#     array[i] = 38.0
# rate_of_increase = (39 - 38) / 4
# for i in range(93, 97):
#     temperature = 38 + (i - 92) * rate_of_increase
#     array[i] = round(temperature, 1)  # 保留一位小数
# for i in range(97, 109):
#     array[i] = 39.0
# rate_of_increase = (39.5 - 39) / 4
# for i in range(109, 113):
#     temperature = 39 + (i - 108) * rate_of_increase
#     array[i] = round(temperature, 1)  # 保留一位小数
# for i in range(113, 123):
#     array[i] = 39.5
# rate_of_increase = (40 - 39.5) / 2
# for i in range(123, 125):
#     temperature = 39.5 + (i - 122) * rate_of_increase
#     array[i] = round(temperature, 1)  # 保留一位小数
# for i in range(125, 135):
#     array[i] = 40.0
# rate_of_increase = (40 - 40) / 4
# for i in range(135, 139):
#     temperature = 40 + (i - 134) * rate_of_increase
#     array[i] = round(temperature, 1)  # 保留一位小数
# for i in range(139, 149):
#     array[i] = 40.0
# input_file = "all2_houyi1h.txt"
# output_file = "all2_hum2.txt"
#
# with open(input_file, 'r') as file_in:
#     with open(output_file, 'w') as file_out:
#         for line in file_in:
#             parts = line.split(",")  # 按空格分割每行的内容
#             address = parts[0]  # 地址部分
#             time = int(parts[1])  # 时间标签部分
#             temperature = array[time]  # 转换为温度
#             new_line = f"{address},{temperature}\n"  # 新的一行内容
#             file_out.write(new_line)  # 写入新文件



# import matplotlib.pyplot as plt
# from matplotlib import rcParams
# # 设置全局字体大小
# rcParams.update({'font.size': 25})
# # 假设的时间数据，你需要根据实际数据进行替换
# time_data = [0,15,19,35,41,59,71,81,85,97,101,111,113,123,127,137,143,155]
# # 假设的干球温度数据
# temperature_data = [38,38, 40,40, 42,42, 46,46, 48, 48, 52,52, 54,54, 58,58, 62,62]
# # 假设的湿球温度数据
# humidity_data = [38,38, 39,39, 37,37, 38,38, 39,39, 39.5, 39.5, 40,40, 40,40, 41,41]
#
# # 创建图表
# plt.figure(figsize=(10, 8))
#
# # 绘制温度曲线
# plt.plot(time_data, temperature_data, marker='o', color='red', label='Temperature (℃)')
#
# # 绘制湿度曲线，使用相同的y轴
# plt.plot(time_data, humidity_data, marker='x', color='blue', label='Humidity (%)')
#
# # 添加图例
# plt.legend()
#
# # 添加标题和轴标签
# # plt.title('Tobacco Curing Oven Temperature and Humidity Over Time')
# plt.xlabel('Time (hours)')
# plt.ylabel('Temperature (℃) / Humidity (%)')
#
# # 自动调整y轴范围以适应温度和湿度数据
# plt.ylim(min(min(temperature_data), min(humidity_data))-1, max(max(temperature_data), max(humidity_data))+1)
#
# # 优化布局并显示图表
# plt.tight_layout()
# plt.show()

#MAE
import matplotlib.pyplot as plt
from matplotlib import rcParams
# 设置全局字体大小
rcParams['font.sans-serif']=['SimHei']  # 中文显示
plt.rcParams['svg.fonttype'] = 'none'
rcParams.update({'font.size': 25})
# 假设的时间数据，你需要根据实际数据进行替换
# time_data = [48,64,80,96,112]
time_data = [2,4,6,8]
# temperature_data = [1.41,0.84,1.29,1.15,1.49]
temperature_data = [0.89,1.17,0.84,1.04]
# humidity_data = [0.94,0.63,0.53,0.28,0.69]
humidity_data = [0.87,0.59,0.28,0.60]
# 创建图表
plt.figure(figsize=(10, 8))
# 绘制温度曲线
plt.plot(time_data, temperature_data, marker='o', color='red', label='温度')
# 绘制湿度曲线，使用相同的y轴
plt.plot(time_data, humidity_data, marker='x', color='blue', label='湿度')
# 添加图例
plt.legend()
# plt.xlabel('Patchsize')
plt.xlabel('Kblock')
plt.ylabel('MAE', rotation=0, verticalalignment='top', horizontalalignment='right')
# plt.xticks([48,64,80,96,112])
# plt.yticks([0,0.25,0.5,0.75,1,1.25,1.5,1.75])
plt.xticks([2,4,6,8])
plt.yticks([0.25,0.5,0.75,1,1.25,1.5])
# 优化布局并显示图表
plt.tight_layout()
# plt.show()
plt.savefig('./emf_pic/svg/kblock.svg',format='svg',dpi=150)
# plt.savefig('./emf_pic/svg/Patchsize.svg',format='svg',dpi=150)




