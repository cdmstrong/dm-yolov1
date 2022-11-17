import torch
import torch.nn as nn
class Loss_yolov1(nn.Module):
    def __init__(self):
        super(Loss_yolov1,self).__init__()

    def forward(self, pred, labels):
        """
        :param pred: (batchsize,30,7,7)的网络输出数据
        :param labels: (batchsize,30,7,7)的样本标签数据
        :return: 当前批次样本的平均损失
        """
        
        num_gridx, num_gridy = labels.size()[-2:]  # 划分网格数量
        num_b = 2  # 每个网格的bbox数量
        num_cls = 20  # 类别数量
        noobj_confi_loss = 0.  # 不含目标的网格损失(只有置信度损失)
        coor_loss = 0.  # 含有目标的bbox的坐标损失
        obj_confi_loss = 0.  # 含有目标的bbox的置信度损失
        class_loss = 0.  # 含有目标的网格的类别损失
        n_batch = labels.size()[0]  # batchsize的大小

        # 可以考虑用矩阵运算进行优化，提高速度，为了准确起见，这里还是用循环
        for i in range(n_batch):  # batchsize循环
            for n in range(7):  # x方向网格循环
                for m in range(7):  # y方向网格循环
                    if labels[i,4,m,n]==1:# 如果包含物体
                        # 将数据(px,py,w,h)转换为(x1,y1,x2,y2)
                        # 先将px,py转换为cx,cy，即相对网格的位置转换为标准化后实际的bbox中心位置cx,xy
                        # 然后再利用(cx-w/2,cy-h/2,cx+w/2,cy+h/2)转换为xyxy形式，用于计算iou
                        bbox1_pred_xyxy = ((pred[i,0,m,n]+n)/num_gridx - pred[i,2,m,n]/2,(pred[i,1,m,n]+m)/num_gridy - pred[i,3,m,n]/2,
                                           (pred[i,0,m,n]+n)/num_gridx + pred[i,2,m,n]/2,(pred[i,1,m,n]+m)/num_gridy + pred[i,3,m,n]/2)
                        bbox2_pred_xyxy = ((pred[i,5,m,n]+n)/num_gridx - pred[i,7,m,n]/2,(pred[i,6,m,n]+m)/num_gridy - pred[i,8,m,n]/2,
                                           (pred[i,5,m,n]+n)/num_gridx + pred[i,7,m,n]/2,(pred[i,6,m,n]+m)/num_gridy + pred[i,8,m,n]/2)
                        bbox_gt_xyxy = ((labels[i,0,m,n]+n)/num_gridx - labels[i,2,m,n]/2,(labels[i,1,m,n]+m)/num_gridy - labels[i,3,m,n]/2,
                                        (labels[i,0,m,n]+n)/num_gridx + labels[i,2,m,n]/2,(labels[i,1,m,n]+m)/num_gridy + labels[i,3,m,n]/2)
                        iou1 = calculate_iou(bbox1_pred_xyxy,bbox_gt_xyxy)
                        iou2 = calculate_iou(bbox2_pred_xyxy,bbox_gt_xyxy)
                        # 选择iou大的bbox作为负责物体
                        if iou1 >= iou2:
                            coor_loss = coor_loss + 5 * (torch.sum((pred[i,0:2,m,n] - labels[i,0:2,m,n])**2) \
                                        + torch.sum((pred[i,2:4,m,n].sqrt()-labels[i,2:4,m,n].sqrt())**2))
                            obj_confi_loss = obj_confi_loss + (pred[i,4,m,n] - iou1)**2
                            # iou比较小的bbox不负责预测物体，因此confidence loss算在noobj中，注意，对于标签的置信度应该是iou2
                            noobj_confi_loss = noobj_confi_loss + 0.5 * ((pred[i,9,m,n]-iou2)**2)
                        else:
                            coor_loss = coor_loss + 5 * (torch.sum((pred[i,5:7,m,n] - labels[i,5:7,m,n])**2) \
                                        + torch.sum((pred[i,7:9,m,n].sqrt()-labels[i,7:9,m,n].sqrt())**2))
                            obj_confi_loss = obj_confi_loss + (pred[i,9,m,n] - iou2)**2
                            # iou比较小的bbox不负责预测物体，因此confidence loss算在noobj中,注意，对于标签的置信度应该是iou1
                            noobj_confi_loss = noobj_confi_loss + 0.5 * ((pred[i, 4, m, n]-iou1) ** 2)
                        class_loss = class_loss + torch.sum((pred[i,10:,m,n] - labels[i,10:,m,n])**2)
                    else:  # 如果不包含物体
                        noobj_confi_loss = noobj_confi_loss + 0.5 * torch.sum(pred[i,[4,9],m,n]**2)

        loss = coor_loss + obj_confi_loss + noobj_confi_loss + class_loss
        # 此处可以写代码验证一下loss的大致计算是否正确，这个要验证起来比较麻烦，比较简洁的办法是，将输入的pred置为全1矩阵，再进行误差检查，会直观很多。
        return loss/n_batch
    
def calculate_iou(bbox1,bbox2):
    """计算bbox1=(x1,y1,x2,y2)和bbox2=(x3,y3,x4,y4)两个bbox的iou"""
    intersect_bbox = [0., 0., 0., 0.]  # bbox1和bbox2的交集
    if bbox1[2]<bbox2[0] or bbox1[3]<bbox2[1] or bbox1[0]>bbox2[2] or bbox1[1]>bbox2[3] : 
        pass
    else:
        intersect_bbox[0] = max(bbox1[0],bbox2[0])
        intersect_bbox[1] = max(bbox1[1],bbox2[1])
        intersect_bbox[2] = min(bbox1[2],bbox2[2])
        intersect_bbox[3] = min(bbox1[3],bbox2[3])

    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])  # bbox1面积
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])  # bbox2面积
    area_intersect = (intersect_bbox[2] - intersect_bbox[0]) * (intersect_bbox[3] - intersect_bbox[1])  # 交集面积
    # print(bbox1,bbox2)
    # print(intersect_bbox)
    # input()

    if area_intersect>0:
        return area_intersect / (area1 + area2 - area_intersect)  # 计算iou
    else:
        return 0
