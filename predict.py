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
import cv2
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def read_img(img_path):
	im=cv2.imread(img_path,0)
	im1=cv2.imread(img_path,1)
	img=cv2.resize(im,(96,96))
	img=np.expand_dims(img,axis=-1)
	img=np.expand_dims(img,axis=0)
	return im1,img

with tf.Session() as sess:
	saver=tf.train.import_meta_graph('ckpt2/train.ckpt-8.meta')
	saver.restore(sess,tf.train.latest_checkpoint('ckpt2/'))
	graph = tf.get_default_graph()
	x = graph.get_tensor_by_name("x:0")
	logits = graph.get_tensor_by_name("logits_eval:0")
	while True:
		path = input('Input image filename:')
		if path=='n':
			break
		try:
			im,img=read_img(path)
		except:
			print('Open Error! Try again!')
			continue
		feed_dict = {x:img}
		classification_result = sess.run(logits,feed_dict)
		#print(tf.argmax(classification_result,1).eval())
		class_num=tf.argmax(classification_result,1).eval()
		if class_num==0:
			str='Tumor'
		if class_num==1:
			str='Normal'
		print('classification results:', str)
		h,w=im.shape[0],im.shape[1]
		img_class=cv2.putText(im,str,org=(int(h/2),int(w/2)),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0,0,255),thickness=3)
		cv2.imshow(str,img_class)
		cv2.waitKey()
		cv2.destroyAllWindows()
		


	
