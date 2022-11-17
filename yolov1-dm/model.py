import torchvision.models as tvmodel
import torch.nn as nn
import torch
from cfg import *

class YOLOv1_resnet(nn.Module):
    def __init__(self):
        super(YOLOv1_resnet,self).__init__()
        resnet = tvmodel.resnet34(pretrained=True)  # 调用torchvision里的resnet34预训练模型
        resnet_out_channel = resnet.fc.in_features  # 记录resnet全连接层之前的网络输出通道数，方便连入后续卷积网络中
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # 去除resnet的最后两层
        # 以下是YOLOv1的最后四个卷积层
        self.Conv_layers = nn.Sequential(
            nn.Conv2d(resnet_out_channel,1024,3,padding=1),
            nn.BatchNorm2d(1024),  # 为了加快训练，这里增加了BN层，原论文里YOLOv1是没有的
            nn.LeakyReLU(),
            nn.Conv2d(1024,1024,3,stride=2,padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
        )
        # 以下是YOLOv1的最后2个全连接层
        self.Conn_layers = nn.Sequential(
            nn.Linear(7*7*1024,4096),
            nn.LeakyReLU(),
            nn.Linear(4096,7*7*12), #30
            nn.Sigmoid()  # 增加sigmoid函数是为了将输出全部映射到(0,1)之间，因为如果出现负数或太大的数，后续计算loss会很麻烦
        )

    def forward(self, input):
        input = self.resnet(input)
        input = self.Conv_layers(input)
        input = input.view(input.size()[0],-1)
        input = self.Conn_layers(input)
        return input.reshape(-1, (5*NUM_BBOX+len(CLASSES)), 7, 7)  # 记住最后要reshape一下输出数据
if __name__ == '__main__':
    x = torch.randn((1,3,448,448))
    net = YOLOv1_resnet()
    print(net)
    y = net(x)
    print(y.size())