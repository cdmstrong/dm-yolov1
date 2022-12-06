
import torch.nn as nn
import torch.nn.functional as F
import torch 
# 常用的卷积层类
class ConvLayers(nn.Module):
    def __init__(self, in_channels, filters) -> None:
        super().__init__()
        self.cov1 = nn.Conv2d(in_channels= in_channels, out_channels=filters, kernel_size=3, padding='same')
        nn.BatchNorm2d(filters, eps=1e-05)
        
        self.cov2 = nn.Conv2d(in_channels= filters, out_channels=filters, kernel_size=3, padding='same')
        nn.BatchNorm2d(filters, eps=1e-05)
    def forward(self, x):
        out = F.relu(self.cov1(x))
        out = F.relu(self.cov2(out))
        
        return out
# 下采样

class DownSample(nn.Module):
    def __init__(self, size) -> None:
        super().__init__()
        self.pool_size = size
        
    def forward(self, x):
        return nn.MaxPool2d(self.pool_size)(x)
# 上采样
    
class Upsample(nn.Module):
    def __init__(self, in_c, out_c) -> None:
        super().__init__()  
        self.Up = nn.ConvTranspose2d(in_c, in_c // 2, 2, 2)
        self.bn1 = nn.BatchNorm2d(out_c)
        
    def forward(self, x, x2):
        x1 = F.relu(self.bn1(self.Up(x)))
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return x1
class Unet(nn.Module):
    def __init__(self, n_classes = 2) -> None:
        super().__init__()
        self.conv1 = ConvLayers(3, 64)
        self.down1 = DownSample(2)
        
        self.conv2 = ConvLayers(64, 128)
        self.down2 = DownSample(2)
        
        self.conv3 = ConvLayers(128, 256)
        self.down3 = DownSample(2)
        
        self.conv4 = ConvLayers(256, 512)
        self.down4 = DownSample(2)
        
        self.conv5 = ConvLayers(512, 1024)
        
        self.up1 = Upsample(1024, 512)
        self.dConv1 = ConvLayers(1024, 512)
        
        self.up2 = Upsample(512, 256)
        self.dConv2 = ConvLayers(512, 256)
        
        self.up3 = Upsample(256, 128)
        self.dConv3 = ConvLayers(256, 128)
        
        self.up4 = Upsample(128, 64)
        self.dConv4 = ConvLayers(128, 64)
        
        self.dOut = nn.Conv2d(64, n_classes, 1)
    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(self.down1(out1))
        out3 = self.conv3(self.down2(out2))
        out4 = self.conv4(self.down3(out3))
        out5 = self.conv5(self.down4(out4))
        # 上卷积
        up_out1 = self.up1(out5, out4)
        cat1 = torch.cat([out4, up_out1], dim= 1)
        dOut1 = self.dConv1(cat1)
        
        up_out2 = self.up2(dOut1, out3)
        cat2 = torch.cat([out3, up_out2], dim=1)
        dOut2 = self.dConv2(cat2)
        
        up_out2 = self.up3(dOut2, out2)
        cat3 = torch.cat([out2, up_out2], dim=1)
        dOut3 = self.dConv3(cat3)
        
        up_out3 = self.up4(dOut3, out1)
        cat4 = torch.cat([out1, up_out3], dim=1)
        dOut4 = self.dConv4(cat4)
        
        out = self.dOut(dOut4)
        
        # 融合
        out = torch.sigmoid(out)
        
        return out
        
if __name__ == '__main__':
    model = Unet()
    inp = torch.rand(4,3,256,256)
    outp = model(inp)
    print(outp.shape)