import torch
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
import os
import datetime
import time

from randomtest import GlobalLocalBrainAge
def eval_single_image(image_path, model, result_save_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([170, 120], antialias=True),
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)
    file_path, file_name = os.path.split(image_path)
    # parent_path, parent_name = os.path.split(file_path)
    # # print(parent_name,file_name)
    # parent_name = parent_name[:4] + '-' + parent_name[4:6] + '-' + parent_name[6:]
    # a_s = tuple(time.strptime(parent_name, "%Y-%m-%d"))
    # d1 = datetime.date(a_s[0], a_s[1], a_s[2])
    # start = '20230625'
    # start = start[:4] + '-' + start[4:6] + '-' + start[6:]
    # s = tuple(time.strptime(start, "%Y-%m-%d"))
    # d2 = datetime.date(s[0], s[1], s[2])
    # # times=(d1 - d2).days*24+int(file_name[:2])+round(int(file_name[2:4])/60,3)
    # times = (d1 - d2).days * 24 + int(file_name[:2]) - 10
    with torch.no_grad():
        model.eval()
        time_pd = model(image)
    with open(result_save_path, "w", encoding="utf-8-sig") as fp:
        # time_init = times
        total_time = 0
        count = 0
        for k in range(1, len(time_pd)):
            time2 = time_pd[k].flatten(0)
            time2_pd = time2.item() * 144
            total_time += time2_pd
            count += 1
        average_time_j = total_time / count
        # print(time_init,average_time_j)
        # fp.write("{},{}\n".format(
        #     round(time_init),
        #     round(average_time_j)
        # ))
        fp.write("{}\n".format(
            # round(time_init),
            round(average_time_j)
        ))
    average_time_j = round(average_time_j)
    # loss = abs(time_init - average_time_j)
    # print("真实值:{},预测值:{}".format(time_init,average_time_j))
    print("预测值:{}".format(average_time_j))
    # print("loss:{}".format(loss))
# image_path = "/home/llj/code/test/data2_rb/20230628/002405_ch01.jpg"
image_path = "./1.jpg"
result_save_path = "./result.csv"
model = GlobalLocalBrainAge(3, patch_size=64, nblock=6)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
state_dict = torch.load("./middle/models/FFT2_ps96_houyi1h_2-best.pth", map_location=device)
model.load_state_dict(state_dict)
eval_single_image(image_path, model, result_save_path)
