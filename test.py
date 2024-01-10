from models import ResNet_50
import torch
import torch.nn as nn
class SELayer(nn.Module):
    def __init__(self, channel=64, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

if __name__ == "__main__":
    # inputs = torch.randn(16, 64, 16, 16)
    # model = SELayer()

    inputs = torch.randn(16, 4, 16, 16)
    model = ResNet_50()

    a = model(inputs)
    print(a.shape)


