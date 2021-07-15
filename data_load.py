import tensorflow as tf
from skimage import io,transform
import glob
import os
import numpy as np
import time
import matplotlib.pyplot as plt
 
def read_img(path,w,h,c):
     #os.listdir(path) 返回path指定的文件夹包含的文件或文件夹的名字的列表
     #os.path.isdir(path)判断path是否是目录
     #b = [x+x for x in list1 if x+x<15 ]  列表生成式,循环list1，当if为真时，将x+x加入列表b
    cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    imgs=[]
    labels=[]
    for idx,folder in enumerate(cate):
         #glob.glob(s+'*.py') 从目录通配符搜索中生成文件列表         
        for im in glob.glob(folder+'/*.jpg'):
            #输出读取的图片的名称             
            #print('reading the images:%s'%(im))
            #io.imread(im)读取单张RGB图片 skimage.io.imread(fname,as_grey=True)读取单张灰度图片
            #读取的图片  
            img=io.imread(im)
            #skimage.transform.resize(image, output_shape)改变图片的尺寸
            img=transform.resize(img,(w,h,c))
            #将读取的图片数据加载到imgs[]列表中
            imgs.append(img)
            #将图片的label加载到labels[]中，与上方的imgs索引对应
            labels.append(idx)
    #将读取的图片和labels信息，转化为numpy结构的ndarr(N维数组对象（矩阵）)数据信息
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)

train_path='f:/train3/'
test1_path='f:/test1/'
test2_path='f:/test4/' #测试集的路径

w,h,c=(96,96,1)
train_data,train_label=read_img(train_path,w,h,c)
test_data1,test_label1=read_img(test1_path,w,h,c)
test_data2,test_label2=read_img(test2_path,w,h,c) 

np.save('training_data',train_data)
np.save('training_label',train_label)
np.save('test_data1',test_data1)
np.save('test_label1',test_label1)
np.save('test_data2',test_data2)
np.save('test_label2',test_label2)

