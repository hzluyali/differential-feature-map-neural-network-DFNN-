#tensorflow 模块
import tensorflow as tf
#skimage模块下的io transform(图像的形变与缩放)模块
from skimage import io,transform
#glob 文件通配符模块 
import glob
#os 处理文件和目录的模块
import os
#多维数据处理模块
import numpy as np
import time
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True #allocate dynamically

# 设置卷积层
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding = "SAME")

# 设置池化层
def pool(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides = [1,2,2,1],padding = "SAME")

def SE_block(x,ratio):
    shape = x.get_shape().as_list()
    channel_out = shape[3]
    out_shape=int(channel_out/ratio)
    # print(shape)
    with tf.variable_scope("squeeze_and_excitation"):
        # 第一层，全局平均池化层
        squeeze = tf.nn.avg_pool(x,[1,shape[1],shape[2],1],[1,shape[1],shape[2],1],padding = "SAME")
        # 第二层，全连接层
        w_excitation1 = weight_variable([1,1,channel_out,out_shape])
        b_excitation1 = bias_variable([out_shape])
        excitation1 = conv2d(squeeze,w_excitation1) + b_excitation1
        excitation1_output = tf.nn.relu(excitation1)
        # 第三层，全连接层
        w_excitation2 = weight_variable([1, 1, out_shape, channel_out])
        b_excitation2 = bias_variable([channel_out])
        excitation2 = conv2d(excitation1_output, w_excitation2) + b_excitation2
        excitation2_output = tf.nn.sigmoid(excitation2)
        # 第四层，点乘
        excitation_output = tf.reshape(excitation2_output,[-1,1,1,channel_out])
        h_output = excitation_output * x

    return h_output

def DF_block(input):
    flip_img=tf.image.flip_left_right(input)
    diff_img=input-flip_img
    result=tf.concat([input,diff_img],axis=3)
    return result

def conv_image_visual(conv_image,image_weight,image_height,cy,cx,channels):
    #slice off one image ande remove the image dimension
    #original image is a 4d tensor[batche_size,weight,height,channels]
    conv_image = tf.slice(conv_image,(0,0,0,0),(1,-1,-1,-1))
    conv_image = tf.reshape(conv_image,(image_height,image_weight,channels))
    #add a couple of pixels of zero padding around the image
    image_weight += 4
    image_height += 4
    conv_image = tf.image.resize_image_with_crop_or_pad(conv_image,image_height,image_weight)
    conv_image = tf.reshape(conv_image,(image_height,image_weight,cy,cx))
    conv_image = tf.transpose(conv_image,(2,0,3,1))
    conv_image = tf.reshape(conv_image,(1,cy*image_height,cx*image_weight,1))
    return conv_image

def weight_variable(shape): #权重初始化
	return tf.Variable(tf.truncated_normal(shape=shape,stddev=0.1))

def bias_variable(shape): #偏置初始化
	return tf.Variable(tf.constant(0.0,shape=shape))
 
def inference(input_tensor, train, regularizer):
#-----------------------第一层----------------------------  
    with tf.variable_scope('layer1-conv1'):
        #初始化权重conv1_weights为可保存变量，大小为5x5,3个通道（RGB），数量为32个
        conv11_weights=weight_variable([3,3,1,16])
        conv11_biases=bias_variable([16])
        conv11=tf.nn.conv2d(input_tensor,conv11_weights,strides=[1,1,1,1],padding='SAME')
        relu11=tf.nn.relu(tf.nn.bias_add(conv11,conv11_biases)) 
        df11=DF_block(relu11)
        se11=SE_block(df11,ratio=4)

    with tf.variable_scope('layer1-conv2'):
        conv12_weight=weight_variable([3,3,32,32])
        conv12_bias=bias_variable([32])
        conv12=tf.nn.conv2d(se11,conv12_weight,strides=[1,1,1,1],padding='SAME')
        relu12=tf.nn.relu(conv12+conv12_bias)
        df12=DF_block(relu12)
        se12=SE_block(df12,ratio=4)
 
    with tf.name_scope("layer1-pool3"):
        #池化计算，调用tensorflow的max_pool函数，strides=[1,2,2,1]，表示池化边界，2个对一个生成，padding="VALID"表示不操作。
        pool13=tf.nn.max_pool(se12,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
 
#-----------------------第二层----------------------------
    with tf.variable_scope("layer2-conv1"):
         #同上，不过参数的有变化，根据卷积计算和通道数量的变化，设置对应的参数  
        conv21_weights=weight_variable([3,3,64,64])
        conv21_bias=bias_variable([64])
        conv21=tf.nn.conv2d(pool13,conv21_weights,strides=[1,1,1,1],padding='SAME')
        relu21=tf.nn.relu(conv21+conv21_bias)
        se21=SE_block(relu21,ratio=4)
        res21_weights=weight_variable([1,1,32,64])
        res21_bias=bias_variable([64])
        res21=tf.nn.conv2d(se11,res21_weights,strides=[1,1,1,1],padding='SAME')
        res21_relu=tf.nn.relu(res21+res21_bias)
        res21_pool=tf.nn.max_pool(res21_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
        res21_out=res21_pool+se21
        

    with tf.name_scope("layer2-pool2"):
        pool22=tf.nn.max_pool(res21_out,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
 
#-----------------------第三层----------------------------
        #同上，不过参数的有变化，根据卷积计算和通道数量的变化，设置对应的参数
    with tf.variable_scope("layer3-conv1"):
        conv31_weights=weight_variable([3,3,64,64])
        conv31_bias=bias_variable([64])
        conv31=tf.nn.conv2d(pool22,conv31_weights,strides=[1,1,1,1],padding='SAME')
        relu31=tf.nn.relu(conv31+conv31_bias)
        df31=DF_block(relu31)
        se31=SE_block(df31,ratio=4)
 
    with tf.name_scope("layer3-pool2"):
        pool32=tf.nn.max_pool(se31,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
		
#-----------------------第四层----------------------------
        #同上，不过参数的有变化，根据卷积计算和通道数量的变化，设置对应的参数
    with tf.variable_scope("layer4-conv1"):
        conv41_weights=weight_variable([3,3,128,128])
        conv41_bias=bias_variable([128])
        conv41=tf.nn.conv2d(pool32,conv41_weights,strides=[1,1,1,1],padding='SAME')
        relu41=tf.nn.relu(conv41+conv41_bias)
        df41=DF_block(relu41)
        se41=SE_block(df41,ratio=4)
 
    with tf.name_scope("layer4-pool2"):
        pool42=tf.nn.max_pool(se41,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
        #pool5 = tf.nn.max_pool(relu5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
 
#-----------------------第五层----------------------------
        #同上，不过参数的有变化，根据卷积计算和通道数量的变化，设置对应的参数
    with tf.variable_scope("layer5-conv1"):
        conv51_weights=weight_variable([3,3,256,512])
        conv51_bias=bias_variable([512])
        conv51=tf.nn.conv2d(pool42,conv51_weights,strides=[1,1,1,1],padding='SAME')
        relu51=tf.nn.relu(conv51+conv51_bias)

        res51_weights=weight_variable([1,1,128,512])
        res51_bias=bias_variable([512])
        res51=tf.nn.conv2d(pool32,res51_weights,strides=[1,1,1,1],padding='SAME')
        res51_relu=tf.nn.relu(res51+res51_bias)
        res51_pool=tf.nn.max_pool(res51_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
        res51_out=res51_pool+relu51
 
    with tf.name_scope("layer5-pool2"):
        pool52=tf.nn.max_pool(res51_out,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
        pool52=tf.keras.layers.GlobalAveragePooling2D()(pool52)
        
#-----------------------第六层----------------------------
    with tf.variable_scope('layer13-fc3'):
        #同上，不过参数的有变化，根据卷积计算和通道数量的变化，设置对应的参数
        fc3_weights = tf.get_variable("weight", [512, 2],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc3_weights))
        fc3_biases = tf.get_variable("bias", [2], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(pool52, fc3_weights) + fc3_biases
    
     #返回最后的计算结果
    return logit,se11,se12
 
#---------------------------网络结束---------------------------
    