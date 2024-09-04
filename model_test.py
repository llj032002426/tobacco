from models import VGG16
from models import SSRNet
from models import ResNet_50
from models import ResNet18
from models import VGG16
import torch
if __name__ == '__main__':
    x = torch.rand(16, 3, 170, 120)
    # model = SSRNet()
    # model = ResNet_50()
    # model = ResNet18()
    model = VGG16()
    out = model(x)
    print(out.shape)