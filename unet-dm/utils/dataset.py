from torch.utils.data.dataset import Dataset
import numpy as np
import os
import torch
from PIL import Image
from torchvision.transforms import transforms
class dataSet(Dataset):
    def __init__(self, isTrain = True):
        super(Dataset).__init__()
        if isTrain:
            name_path = f'{os.getcwd()}/unet-dm/data/annotations/trainval.txt'
        else:
            name_path = f'{os.getcwd()}/unet-dm/data/annotations/test.txt'
        
        # 获取所有名字
        # self.name_list = [x.rstrip().split(' ')[0] for x in open(name_path, 'r').readlines()]
        self.img_path = f'{os.getcwd()}/unet-dm/data/lgg-mri-segmentation/kaggle_3m/'
        self.masks_path = f'{os.getcwd()}/unet-dm/data/lgg-mri-segmentation/kaggle_3m/'
        self.imgs = []
        self.masks = []
        for file in os.listdir(self.img_path):
            if os.path.isdir(f'{self.img_path}{file}'):
                for dir in os.listdir(f'{self.img_path}{file}'):
                    self.imgs.append(f'{self.img_path}{file}/{dir}') if 'mask' not in dir else self.masks.append(f'{self.img_path}{file}/{dir}')

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        img = Image.open(self.imgs[index])
        mask = Image.open(self.masks[index])
        img = preprocess(img, 1, is_mask=False)
        mask = preprocess(mask, 1, is_mask=True)
        aug = transforms.Compose([
            transforms.ToTensor()
        ])
        img = aug(img)
        mask = torch.tensor(mask)
        
        return img.float().contiguous(), mask.long().contiguous()
def preprocess(pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            # else:
            #     img_ndarray = img_ndarray.transpose((2, 0, 1))
            #img_ndarray = img_ndarray / 255

        return img_ndarray       
if __name__ == "__main__":
    data = dataSet()