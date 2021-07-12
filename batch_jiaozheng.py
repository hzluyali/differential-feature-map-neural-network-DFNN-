import cv2
import numpy as np
import math
import pydicom
import matplotlib.pyplot as plt
import os

def m_f_b(im,threshold): #获取前景背景平均值函数
	#输入图像和阈值
	#输出前景和背景的平均值
	sum=0
	sum1=0
	count=0
	count1=0
	im1=im.copy()
	im2=im.copy()
	count=np.sum(im>=threshold)
	count1=np.sum(im<threshold)
	im1[np.where(im<threshold)]=0
	sum=np.sum(im1)
	im2[np.where(im>=threshold)]=0
	sum1=np.sum(im2)
	# for i in range(im.shape[0]):
		# for j in range(im.shape[1]):
			# if im[i,j]>=threshold:
				# sum=sum+im[i,j]
				# #print(im1[i,j])
				# count=count+1
			# else:
				# sum1=sum1+im[i,j]
				# count1=count1+1
	mean11=sum/count
	mean12=sum1/count1
	return mean11,mean12

	
def iterative_threshold(img): #输入图像，输出图像阈值
	n0=np.min(img) #获取图像的最小值
	n1=np.max(img) #获取图像的最大值
	thre0=(n1+n0)/2 #计算图像的初始阈值
	iteration_time=8 #设置迭代阈值法的迭代次数
	mean1,mean2=m_f_b(img,thre0)
	t0=thre0
	for m in range(iteration_time): #迭代更新阈值
		t=(mean1+mean2)/2
		if abs(t-t0)<=0.001:
			break
		t0=t-t
		mean1,mean2=m_f_b(img,t)
	return t

def segment_brain(img,t): #根据迭代阈值法所求得的阈值去除背景
	im1=img.copy()
	im2=cv2.threshold(im1,t,255,cv2.THRESH_BINARY)[1]
	c=np.where(im2!=0)
	m12=np.max(c[0])   #获取图像背景，前景的分割点，即前景图像的位置
	n12=np.min(c[0])
	m13=np.max(c[1])
	n13=np.min(c[1])
	o=im1[n12:m12,n13:m13] #对各个序列图像进行背景去除
	return o

path='f:/dicom/' #设置存放原始数据的路径
output_path='f:/output/' #设置存放校正后数据的路径
if not os.path.exists(output_path): #如果校正数据路径不存在则创建
	os.mkdir(output_path) 
img_dir=os.listdir(path) #检索该路径下的所有文件名
#print(img_dir)

for i in range(len(img_dir)):
	print(i,img_dir[i])
	# read input (dicom格式文件)
	im1=pydicom.dcmread(path+img_dir[i])
	im2=im1.pixel_array
	n1=np.min(im2)
	m1=np.max(im2)
	im2=(im2-n1)/(m1-n1)
	im2=im2*255.0
	cv2.imwrite('f:/1.jpg',im2)
	img=cv2.imread('f:/1.jpg',1)

	# convert to gray
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# threshold
	t=iterative_threshold(gray)
	thresh = cv2.threshold(gray, t , 255, cv2.THRESH_BINARY)[1]

	# find largest contour
	contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contours = contours[0] if len(contours) == 2 else contours[1]
	big_contour = max(contours, key=cv2.contourArea)

	# fit contour to ellipse and get ellipse center, minor and major diameters and angle in degree 
	ellipse = cv2.fitEllipse(big_contour)
	(xc,yc),(d1,d2),angle = ellipse
	#print(xc,yc,d1,d1,angle)

	if angle<90:
		if angle<=3:
			angle_rotate=angle
		else:
			angle_rotate=3
		# height,width=img.shape[0:2]
		# retval=cv2.getRotationMatrix2D((xc,yc),angle,scale=1) #旋转矩阵
		# rotate_img=cv2.warpAffine(img,retval,(width,height))
	else:
		if abs(angle-180)<=3:
			angle_rotate=angle-180
		else:
			angle_rotate=-3
	height,width=img.shape[0:2]
	retval=cv2.getRotationMatrix2D((xc,yc),angle_rotate,scale=1) #旋转矩阵
	rotate_img=cv2.warpAffine(img,retval,(width,height)) #旋转后图像
	
	## 通过迭代阈值法将旋转后图像进行背景去除
	seg_rotate_img=segment_brain(rotate_img,t)
	cv2.imwrite(output_path+str(i)+'.jpg',seg_rotate_img)

# # draw ellipse
# result = img.copy()
# cv2.ellipse(result, ellipse, (0, 255, 0), 3)

# # draw circle at center
# xc, yc = ellipse[0]
# cv2.circle(result, (int(xc),int(yc)), 10, (255, 255, 255), -1)

# # draw vertical line
# # compute major radius
# rmajor = max(d1,d2)/2
# if angle > 90:
    # angle = angle - 90
# else:
    # angle = angle + 90
# print(angle)
# xtop = xc + math.cos(math.radians(angle))*rmajor
# ytop = yc + math.sin(math.radians(angle))*rmajor
# # print(xtop,ytop)
# xbot = xc + math.cos(math.radians(angle+180))*rmajor
# ybot = yc + math.sin(math.radians(angle+180))*rmajor
# cv2.line(result, (int(xtop),int(ytop)), (int(xbot),int(ybot)), (0, 0, 255), 3)
# cv2.circle(result, (int(xtop),int(ytop)), 10, (255, 0, 0), 3)

# cv2.imwrite("labrador_ellipse.jpg", result)

# cv2.imshow("labrador_thresh", thresh)
# cv2.imshow("labrador_ellipse", result)
# cv2.imshow('rotate',rotate_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
	
