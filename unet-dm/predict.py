
from PIL import Image
import os
from torchvision.transforms import transforms
from model import *
import cv2
import numpy as np

def _fast_hist(label_true, label_pred, n_class):
    """
    label_true是转化为一维数组的真实标签，label_pred是转化为一维数组的预测结果，n_class是类别数
    hist是一个混淆矩阵
    hist是一个二维数组，可以写成hist[label_true][label_pred]的形式
    最后得到的这个数组的意义就是行下标表示的类别预测成列下标类别的数量
    """
    # mask在和label_true相对应的索引的位置上填入true或者false
    # label_true[mask]会把mask中索引为true的元素输出
    mask = (label_true >= 0) & (label_true < n_class)
    # n_class * label_true[mask].astype(int) + label_pred[mask]计算得到的是二维数组元素
    # 变成一位数组元素的时候的地址取值(每个元素大小为1)，返回的是一个numpy的list
    # np.bincount()会给出索引对应的元素个数
    hist = np.bincount(n_class * label_true[mask].astype(int) + label_pred[mask],
                       minlength=n_class ** 2).reshape(n_class, n_class)
    return hist
def per_class_iu(hist):
    # 矩阵的对角线上的值组成的一维数组/(矩阵的每行求和+每列求和-对角线上的值)，返回值形状(n,)
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

if __name__ == '__main__':
    img = Image.open(f'{os.getcwd()}/unet-dm/test.tif')
    mask = Image.open(f'{os.getcwd()}/unet-dm/mask.tif')
    toTensor = transforms.ToTensor()
    img1 = toTensor(img).to('cpu')
    
    model = Unet()
    print(f'{os.getcwd()}/unet-dm/models_pkl/unet1.pkl')
    model = torch.load(f'{os.getcwd()}/unet-dm/models_pkl/unet1.pkl', map_location='cpu')
    # model.load_state_dict(dist, strict=True)
    model.eval()
    with torch.no_grad():
        out = model(img1.unsqueeze(0))
        out = F.softmax(out.squeeze(0).permute(2, 1, 0), dim=-1)
        out_mask = (out.numpy())*255
        out_binary = np.argmax(out_mask, axis=2)
        out_mask = Image.fromarray(np.uint8(out_mask)).convert('RGB')
        out_mask.show()
        
        miou = per_class_iu(_fast_hist(np.uint8(mask), out_binary, 2))
        out_binary = Image.fromarray(np.uint8(out_binary)).convert('RGB')
        
        out_mask = Image.blend(img, out_binary, alpha = 0.5)
        # out = image1 * (1.0 - alpha) + image2 * alpha
        out_binary.show()

    