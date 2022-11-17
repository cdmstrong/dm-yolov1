import os

from cfg import *
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader
import cv2
import numpy as np
import torchvision.transforms as transforms
class Dataset(data.Dataset):
    def __init__(self, is_train = True, is_aug = True) -> None:
        """
        :param is_train: 调用的是训练集(True)，还是验证集(False)
        :param is_aug:  是否进行数据增广
        """
        super().__init__()
        self.root = 'f:\python\dm-net\yolov1-dm/yolo_helmet_train/anno/'
        self.img_root = self.root + "images/"
        self.labels_root = self.root + "labels/"
        self.train_txt = "train.txt"
        
        if is_train:
            with open(self.root + self.train_txt, 'r') as f:
                self.file_name = [line.strip() for line in f.readlines()]
        else:
            with open(self.root + self.valid_txt, 'r') as f:
                self.file_name = [line.strip() for line in f.readlines()]
        # 图片地址
        self.is_aug = is_aug
    def __getitem__(self, item):
        img = cv2.imread(self.img_root + self.file_name[item] + ".jpg")
        h, w = img.shape[0:2]
        input_size = 448 #v1的输入大小
        # 因为数据集内原始图像的尺寸是不定的，所以需要进行适当的padding，将原始图像padding成宽高一致的正方形
        # 然后再将Padding后的正方形图像缩放成448x448
        padw, padh = 0, 0 # 要记录宽高方向的padding具体数值，因为padding之后需要调整bbox的位置信息
        if h>w:
            padw = (h - w) // 2
            img = np.pad(img,((0,0),(padw,padw),(0,0)),'constant',constant_values=0)
        elif w>h:
            padh = (w - h) // 2
            img = np.pad(img,((padh,padh),(0,0),(0,0)), 'constant', constant_values=0)
        img = cv2.resize(img,(input_size, input_size))
        # 图像增广部分，这里不做过多处理，因为改变bbox信息还蛮麻烦的
        if self.is_aug:
            aug = transforms.Compose([
                transforms.ToTensor()
            ])
            img = aug(img)
        bbox = None
        # 读取图像对应的bbox信息，按1维的方式储存，每5个元素表示一个bbox的(cls,xc,yc,w,h)
        try:
            with open(self.labels_root + self.file_name[item] + ".txt") as f:
                bbox = f.read().split('\n')
        except  Exception as e :
            print('input idx is{}'.format(item))
        finally:
            pass
        bbox = [x.split() for x in bbox]
        bbox = [float(x) for y in bbox for x in y]
        if len(bbox)%5!=0:
            raise ValueError("File:"+self.labels_root+self.file_name[item]+".txt"+"——bbox Extraction Error!")

        # 根据padding、图像增广等操作，将原始的bbox数据转换为修改后图像的bbox数据
        for i in range(len(bbox)//5):
            if padw != 0:
                bbox[i * 5 + 1] = (bbox[i * 5 + 1] * w + padw) / h
                bbox[i * 5 + 3] = (bbox[i * 5 + 3] * w) / h
            elif padh != 0:
                bbox[i * 5 + 2] = (bbox[i * 5 + 2] * h + padh) / w
                bbox[i * 5 + 4] = (bbox[i * 5 + 4] * h) / w
            # 此处可以写代码验证一下，查看padding后修改的bbox数值是否正确，在原图中画出bbox检验

        labels = convert_bbox2labels(bbox)  # 将所有bbox的(cls,x,y,w,h)数据转换为训练时方便计算Loss的数据形式(7,7,5*B+cls_num)
        # 此处可以写代码验证一下，经过convert_bbox2labels函数后得到的labels变量中储存的数据是否正确
        labels = transforms.ToTensor()(labels)
        return img, labels
    def __len__(self):
        return len(self.file_name)
def convert_bbox2labels(bbox):
    """将bbox的(cls,x,y,w,h)数据转换为训练时方便计算Loss的数据形式(7,7,5*B+cls_num)
    注意，输入的bbox的信息是(xc,yc,w,h)格式的，转换为labels后，bbox的信息转换为了(px,py,w,h)格式"""
    gridsize = 1.0/7
    labels = np.zeros((7,7,5*NUM_BBOX+len(CLASSES)))  # 注意，此处需要根据不同数据集的类别个数进行修改
    for i in range(len(bbox)//5):
        gridx = int(bbox[i*5+1] // gridsize)  # 当前bbox中心落在第gridx个网格,列
        gridy = int(bbox[i*5+2] // gridsize)  # 当前bbox中心落在第gridy个网格,行
        # (bbox中心坐标 - 网格左上角点的坐标)/网格大小  ==> bbox中心点的相对位置
        gridpx = bbox[i * 5 + 1] / gridsize - gridx
        gridpy = bbox[i * 5 + 2] / gridsize - gridy
        # 将第gridy行，gridx列的网格设置为负责当前ground truth的预测，置信度和对应类别概率均置为1
        labels[gridy, gridx, 0:5] = np.array([gridpx, gridpy, bbox[i * 5 + 3], bbox[i * 5 + 4], 1])
        labels[gridy, gridx, 5:10] = np.array([gridpx, gridpy, bbox[i * 5 + 3], bbox[i * 5 + 4], 1])
        labels[gridy, gridx, 10+int(bbox[i*5])] = 1
    return labels


def show_labels_img(imgname):
	# """imgname是输入图像的名称，无下标"""
    img = cv2.imread(DATASET_PATH + "images/" + imgname + ".jpg")
    h, w = img.shape[:2]
    label = []
    with open(DATASET_PATH + "labels/"+imgname+".txt",'r') as flabel:
        for label in flabel:
            label = label.split(' ')
            label = [float(x.strip()) for x in label]
            pt1 = (int(label[1] * w - label[3] * w / 2), int(label[2] * h - label[4] * h / 2))
            pt2 = (int(label[1] * w + label[3] * w / 2), int(label[2] * h + label[4] * h / 2))
            cv2.putText(img,CLASSES[int(label[0])],pt1,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))
            cv2.rectangle(img,pt1,pt2,(0,0,255,2))

    cv2.imshow("img",img)
    cv2.waitKey(0)
# 处理文件名称
def handlerFile():
    i = 0
    out_name = "train"
    train = (len(os.listdir(DATASET_PATH + 'images')) * 0.7)
    for f in os.listdir(DATASET_PATH + 'images'):
        if i > train:
            out_name = 'valid'
        file_name = f.split('.')[0]
        with open(DATASET_PATH + out_name + '.txt', "a") as w:
            w.write(file_name + "\n")
        i += 1
if __name__ == '__main__':
    # show_labels_img("PartB_02333")
    dataset = Dataset()
    # handlerFile()
    loader = DataLoader(dataset = dataset,batch_size = 16,shuffle = True,num_workers = 4)
    iter(loader).next()
    
        