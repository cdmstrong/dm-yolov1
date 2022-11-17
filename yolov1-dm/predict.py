from torch.utils.data import DataLoader
from dataset import *
import torch
from utils import *
if __name__ == '__main__':
    val_dataloader = DataLoader(Dataset(is_train=False), batch_size=1, shuffle=False)
    model = torch.load("./models_pkl/YOLOv1_epoch40.pkl")  # 加载训练好的模型
    for i,(inputs,labels) in enumerate(val_dataloader):
        inputs = inputs.cuda()
        # 以下代码是测试labels2bbox函数的时候再用
        # labels = labels.float().cuda()
        # labels = labels.squeeze(dim=0)
        # labels = labels.permute((1,2,0))
        pred = model(inputs)  # pred的尺寸是(1,30,7,7)
        pred = pred.squeeze(dim=0)  # 压缩为(30,7,7)
        pred = pred.permute((1,2,0))  # 转换为(7,7,30)

        ## 测试labels2bbox时，使用 labels作为labels2bbox2函数的输入
        bbox = labels2bbox(pred)  # 此处可以用labels代替pred，测试一下输出的bbox是否和标签一样，从而检查labels2bbox函数是否正确。当然，还要注意将数据集改成训练集而不是测试集，因为测试集没有labels。
        inputs = inputs.squeeze(dim=0)  # 输入图像的尺寸是(1,3,448,448),压缩为(3,448,448)
        inputs = inputs.permute((1,2,0))  # 转换为(448,448,3)
        img = inputs.cpu().numpy()
        img = 255*img  # 将图像的数值从(0,1)映射到(0,255)并转为非负整形
        img = img.astype(np.uint8)
        draw_bbox(img,bbox.cpu())  # 将网络预测结果进行可视化，将bbox画在原图中，可以很直观的观察结果
        print(bbox.size(),bbox)
        input()
