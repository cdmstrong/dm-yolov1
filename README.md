# 本項目鏈接地址：https://github.com/cdmstrong/dm-yolov1
## 作者： 陈-dm

* 数据集下载地址：https://pan.baidu.com/s/1m1ysN5r0wYlIUY6FW1XkHg#list/path=%2F
* 头盔的训练集的数据集下载地址(百度网盘 Password: y694 )

## 更多训练数据参考项目：
* https://codechina.csdn.net/EricLee/yolo_v3

* 效果展示：
[image](https://codechina.csdn.net/EricLee/yolo_v3/-/raw/master/samples/hat.gif)

* yolov1 论文参考：
https://arxiv.org/pdf/1506.02640.pdf

* 参考博客：
https://blog.csdn.net/weixin_43702653/article/details/123959840


# 本项目环境

pytorch
numpy

# 运行说明

## 数据处理
* 运行dataset.py 下的handlerFile方法
```
python handlerFile
```
将train.txt 分成train 和 valid ， 默认是0.3的测试集

## 训练 
运行 train.py
* 需要修改你所要的种类， 目前为两类， helmet 和no-helmet
* model.py 的全连接层需要改成你要训练的种类数量

## 预测 
运行 predict.py

## 文件说明
* cfg  配置文件
* dataset 数据处理
* loss 损失函数
* model 模型文件
* predict 预测文件
* train 训练文件
* utils 工具文件

