# differential-feature-map-neural-network-DFNN-For-Brain-Tumor-Recognition
Tensorflow version of DFNN，准确率超过99%，网络参数量仅2.26M

## 所需库：Tensorflow 1.13+Numpy+Skimage+Matplotlib+OpenCV+pydicom

## 文件功能

jiaozheng.py #对单幅图像进行图像校正,输入dicom格式文件，输出校正后的图像，保存为jpg格式

batch_jiaozheng.py #对多幅图像进行批量的图像校正

data_load.py #读取数据，并将数据转化为.npy格式文件，方便调用

model.py #存放DFNN模型的结构及参数

train.py #训练模型

predict.py #利用已训练数据预测新图片类别并可视化

## 模型搭建步骤 (训练自己的模型)：

Step 1 (可选): 执行 python batch_jiaozheng.py实现自动图像校正，使得脑MRI影像的对称轴与铅垂线平移，总体思路及效果如下所示：

![Image text](https://github.com/hzluyali/differential-feature-map-neural-network-DFNN-/blob/main/img/1626449101(1).jpg)

注意：不进行图像校正也可获得较好的结果！

Step 2: 执行python data_load.py，将图像数据转化为.npy格式，便于存取，训练自己数据集时需修改data_load.py中33-35行的文件路径

Step 3: 执行python train.py训练DFNN模型

## 模型搭建步骤 (直接调用已训练好的模型)：
将模型文件放置于ckpt2文件夹中

执行python predict.py 预测新图片，需修改第25行和第26行的模型文件路径，此外可以将图像进行可视化

![Image text](https://github.com/hzluyali/differential-feature-map-neural-network-DFNN-/blob/main/img/1626452647(1).jpg)



