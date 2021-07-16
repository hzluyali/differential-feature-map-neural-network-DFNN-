# differential-feature-map-neural-network-DFNN-For-Brain-Tumor-Recognition
Tensorflow version of DFNN

Tensorflow 1.13+Numpy+Skimage+Matplotlib

jiaozheng.py #对单幅图像进行图像校正,输入dicom格式文件，输出校正后的图像，保存为jpg格式

batch_jiaozheng.py #对多幅图像进行批量的图像校正

data_load.py #读取数据，并将数据转化为.npy格式文件，方便调用

model.py #存放DFNN模型的结构及参数

train.py #训练模型

模型搭建步骤：
Step 1 (可选): 执行 python batch_jiaozheng.py实现自动图像校正，使得脑MRI影像的对称轴与铅垂线平移，总体思路及效果如下所示：


