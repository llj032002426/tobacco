# import re
#
# # 输入文件和输出文件的路径
# input_file = "./middle/result/llj2.csv"
# output_file = "output.txt"
#
# # 正则表达式模式
# pattern = r'/.*?(\d+),(\d+)\s*$'
#
# # 保存匹配行的列表
# matched_lines = []
#
# # 打开输入文件
# with open(input_file, 'r') as fin:
#     # 遍历输入文件的每一行
#     for line in fin:
#         # 匹配模式
#         match = re.search(pattern, line)
#
#         if match:
#             # 提取两个数字
#             num1 = int(match.group(1))
#             num2 = int(match.group(2))
#
#             # 检查差异是否大于4
#             if abs(num1 - num2) <= 6:
#                 # 将匹配的行添加到列表中
#                 matched_lines.append(line)
#
# # 根据每行的第一个数字进行排序
# matched_lines.sort(key=lambda x: int(re.search(pattern, x).group(1)))
#
# # 将排序后的结果写入到输出文件
# with open(output_file, 'w') as fout:
#     fout.writelines(matched_lines)
#
# print("筛选并排序完成，结果已写入到", output_file)

# import matplotlib.pyplot as plt
# # pathsize
# x=[32,48,64,80]
# y=[6.04,2.71,2.13,5.56]
# plt.plot(x, y)
# plt.xticks([32,48,64,80])
# plt.yticks([2, 3.5, 5, 6.5,8])
# plt.xlabel('pathsize')
# plt.ylabel('MAE')
# plt.show()
#
# # nblock
# x=[2,4,6,8,10]
# y=[8.55,7.2,2.13,7.23,10.46]
# plt.plot(x, y)
# plt.xticks([2,4,6,8,10])
# plt.yticks([2, 5,7,9,12])
# plt.xlabel('nblock')
# plt.ylabel('MAE')
# plt.show()


# count = 0
# total = 0
# loss = 0
# with open("./results/best.txt", "r", encoding="utf-8-sig") as f:
#     f.readline()
#     for line in f:
#         total += 1
#         filename, time_gt, time_pd = line.strip().split(",")
#         # loss += (int(time_gt)/100.0 - int(time_pd)/100.0)**2
#         # loss += (int(time_gt) - int(time_pd)) ** 2
#         loss += abs(int(time_gt) - int(time_pd))
#         # loss += (float(time_gt) / 100.0 - float(time_pd) / 100.0) ** 2
#         # if int(time_gt) == int(time_pd) :
#         if abs(int(time_gt) - int(time_pd)) <= 4:
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


# def process_txt_file(input_file, output_file):
#     with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
#         for line in f_in:
#             # 分割每行内容
#             parts = line.strip().split()
#             # 提取epoch、trainloss和valloss
#             epoch = parts[4].strip()
#             epoch = epoch[7:]
#             train_loss = parts[8].strip()
#             train_loss = train_loss[5:]
#             val_loss = parts[9].strip()
#             val_loss = val_loss[9:]
#             # 写入到输出文件
#             f_out.write(f"{epoch},{train_loss},{val_loss}\n")
#
# # 请替换为你的输入文件路径和输出文件路径
# input_file_path = 'loss.txt'
# output_file_path = './results/nblock10_loss.txt'
# process_txt_file(input_file_path, output_file_path)



# import matplotlib.pyplot as plt
# def plot_loss_from_file(loss_file):
#     epochs = []
#     train_losses = []
#     val_losses = []
#
#     with open(loss_file, 'r') as file:
#         for line in file:
#             epoch, train_loss, val_loss = map(float, line.strip().split(","))
#             epochs.append(epoch)
#             train_losses.append(train_loss)
#             val_losses.append(val_loss)
#
#     plt.plot(epochs[0:], train_losses[0:], label='Train Loss')
#     plt.plot(epochs[0:], val_losses[0:], label='Validation Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Training and Validation Losses')
#     plt.legend()
#     # plt.grid(True)
#     plt.show()
#
# # 请替换为你的训练结果损失文件路径
# loss_file_path = './results/03240904.txt'
# plot_loss_from_file(loss_file_path)


def process_txt_file(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            # 分割每行内容
            parts = line.strip().split(',')
            # 提取数字1和数字2
            num1 = int(parts[1])
            num2 = int(parts[2])
            # 计算两个数字的差的绝对值
            result = str(abs(num1 - num2))
            # 将结果写入输出文件
            f_out.write("{}\n".format(
                result
            ))

# 请替换为你的输入文件路径和输出文件路径
input_file_path = './results/Resnet50.csv'
output_file_path = './results/Resnet50_pd.txt'
# process_txt_file(input_file_path, output_file_path)



import matplotlib.pyplot as plt

def calculate_cs_scores(input_files):
    cs_scores_data = {}

    for input_file in input_files:
        numbers = []

        # 读取文件并将数字添加到列表中
        with open(input_file, 'r') as f:
            for line in f:
                number = int(line.strip())
                numbers.append(number)

        # 对数字列表进行排序
        sorted_numbers = sorted(numbers)
        total_samples = len(sorted_numbers)

        # 计算不超过每个阈值的数字的个数，并计算CS(α)
        cs_scores = []
        thresholds = range(6)  # α从0到6
        for threshold in thresholds:
            count_below_threshold = sum(1 for num in sorted_numbers if num <= threshold)
            cs_score = count_below_threshold / total_samples * 100
            cs_scores.append(cs_score)

        cs_scores_data[input_file] = cs_scores

    return thresholds, cs_scores_data

def plot_cs_curves(thresholds, cs_scores_data):
    for input_file, cs_scores in cs_scores_data.items():
        label = input_file[10:]
        label = label[:-7]
        plt.plot(thresholds, cs_scores, linestyle='-', label=label)

    plt.xlabel('Threshold (alpha)')
    plt.ylabel('Cumulative Score Score (%)')
    # plt.title('CS Cumulative Score Curve')
    # plt.grid(True)
    plt.xticks(range(6))  # 设置x轴刻度为0到6
    plt.yticks(range(0, 101, 10))  # 设置y轴刻度为0到100，间隔为10
    plt.legend()
    plt.show()

# 请替换为你的输入文件路径列表
input_files = ['./results/Resnet50_pd.txt', './results/SSRnet_pd.txt', './results/Global-Local Transformer_pd.txt']

# 计算CS(α)并绘制CS曲线对比
# thresholds, cs_scores_data = calculate_cs_scores(input_files)
# plot_cs_curves(thresholds, cs_scores_data)


def metrics():
    count = 0
    total = 0
    loss = 0
    with open("./results/1.csv", "r", encoding="utf-8-sig") as f:
        f.readline()
        for line in f:
            total += 1
            filename, time_gt,time_pd= line.strip().split(",")
            # loss += (int(time_gt)/100.0 - int(time_pd)/100.0)**2
            # loss += (int(time_gt) - int(time_pd)) ** 2
            loss += abs(int(time_gt) - int(time_pd))
            # loss += (float(time_gt) / 100.0 - float(time_pd) / 100.0) ** 2
            # if int(time_gt) == int(time_pd) :
            if abs(int(time_gt) - int(time_pd))<=4 :
            # time_gt = int(time_gt)
            # time_pd = int(time_pd)
            # if Acc(time_gt, time_pd):
                count +=1
    print("loss:{}, acc:{:.6f}".format(loss/total, count/total))  # 此处离线评估的loss会比训练期间的验证集更小 因为保存csv时用round做了四舍五入取整
    # print("loss:{}".format(loss / total))
# metrics()

import cv2
import matplotlib.pyplot as plt
import numpy as np
img0 = cv2.imread('/home/llj/code/test/data_rb/20230615/122456_ch01.jpg')
img1 = cv2.resize(img0, dsize=None, fx=0.5, fy=0.5)
img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
plt.imshow(img2)
plt.show()
# plt.rcParams['font.family'] = 'Arial Unicode MS'
f = np.fft.fft2(img2)
fshift = np.fft.fftshift(f)  # 将0频率分量移动到中心
magnitude_spectrum = 20 * np.log(np.abs(fshift))
plt.xticks([])  # 除去刻度线
plt.yticks([])
# plt.title("频谱图")
plt.imshow(magnitude_spectrum, cmap='gray')
plt.show()










