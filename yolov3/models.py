import torchvision.models as tvmodel
import torch.nn as nn
import torch
from utils.parse_config import parse_model_config
from torchsummary import summary


class Darknet(nn.Module):
    def __init__(self, config_path, img_size=416) -> None:
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        model = create_model(self.module_defs)
        
class EmptyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

class YoloLayer(nn.Module):
    def __init__(self, num_classes, anchors, img_size = 416) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.ignore_thres = 0.5
        self.mes_loss = nn.MSELoss() # 位置loss mse
        self.bce_loss = nn.BCELoss() # 二分类loss
        self.obj_scale = 1 
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_size
        self.grid_size = 0  # 切分的网格大小
    def forward(self, x, targets=None, img_dim=None):
        print(x.shape)
        
        self.img_dim = img_dim # 大小可变的
        num_samples = x.size(0) # 输入为(batch_size, channels, img_size, img_size)
        grid_size = x.size(2) # 网格大小 13 * 13
        FloatTensor = torch.FloatTensor
        # 转换数据为特定格式 
        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )
        tx = torch.sigmoid(prediction[..., 0])
        ty = torch.sigmoid(prediction[..., 1])
        tw = prediction[..., 2]
        th = prediction[..., 3]
        conf = torch.sigmoid(prediction[..., 4])
        classes = torch.sigmoid(prediction[..., 5:])
        
        # 因为有三种不同的网格大小， 需要在每次运行后判断网格是否变化，变化需要重新计算
        if img_dim != self.img_dim:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda) #相对位置得到对应的绝对位置比如之前的位置是0.5,0.5变为 11.5，11.5这样的
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = tx.data + self.grid_x
        pred_boxes[..., 1] = ty.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(tw.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(th.data) * self.anchor_h
        # 还原到原始图像
        out_put = torch.cat(
            pred_boxes.view(num_samples, 1, 4) * self.stride,
            conf.view(num_samples, -1, 1),
            classes.view(num_samples, -1, self.num_classes),
        )
        if not targets:
            return out_put, 0
        
    def compute_grid_offsets(self, grid_size, cuda):
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size # 416 / 13 = 32, 说明特征图缩放了32倍
        # 网格大小
        self.scaled_anchors = FloatTensor([(a_w/ self.stride, a_h/self.stride) for a_w, a_h in self.anchors])
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor) 
        
        self.anchor_w = self.scaled_anchors[:, 0:1].view(1, self.num_anchors, 1, 1)
        self.anchor_h = self.scaled_anchors[:, 0:1].view(1, self.num_anchors, 1, 1)

def create_model(module_defs):
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()
    for idx, item in enumerate(module_defs):
        print(item)
        modules = nn.Sequential()
        if item['type'] == 'convolutional':
            bn = int(item['batch_normalize'])
            filters = int(item['filters'])
            size = int(item['size'])
            stride = int(item['stride'])
            pd = (size - 1) //2
            modules.add_module(f'conv_{idx}', nn.Conv2d(
                in_channels=output_filters[-1],
                out_channels=filters,
                kernel_size=size,
                stride=stride,
                padding=pd,
                bias= not bn,
            ))
            if bn:
                modules.add_module(f'batch_norm_{idx}', nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
            if item['activation']:
                modules.add_module(f"leaky_{idx}", nn.LeakyReLU(0.1))
        elif item['type'] == 'shortcut':
            # 残差链接层
            filters = output_filters[1:][int(item['from'])]
            module_list.add_module(f'shortcut_{idx}', EmptyLayer())
        elif item['type'] == 'upsample':
            # 上采样
            upsample = nn.Upsample(scale_factor=int(item['stride']))
            module_list.add_module(f'upsample_{idx}', upsample)
        elif item['type'] == 'route': 
            # 特征拼接金字塔层
            # 获取所有的层数
            layers = [int(x) for x in item['layers'].split(',')]
            filters = sum([output_filters[1:][x] for x in layers])
            module_list.add_module(f'route_{idx}', EmptyLayer())
        elif item['type'] == 'yolo':
            anchors = [int(x) for x in item['anchors'].split(',')]
            anchors_idx = [int(x) for x in item['mask'].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchors_idx]
            num_classes = int(item['classes'])
            img_size = int(hyperparams["height"])
            yolo_layer = YoloLayer(anchors, num_classes, img_size)
            modules.add_module(f"yolo_{idx}", yolo_layer)
            
        output_filters.append(filters)
        module_list.append(modules)
if __name__ == "__main__":
    darknet = Darknet('./config/yolov3.cfg')
    print(darknet)