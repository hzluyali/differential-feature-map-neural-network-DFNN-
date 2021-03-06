import cv2
import numpy as np
import math
import pydicom
import matplotlib.pyplot as plt

def m_f_b(im,threshold): #获取前景背景函数
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
	im2=cv2.threshold(im1,t,cv2.THRESH_BINARY)[1]
	c=np.where(im2!=0)
	m12=np.max(c[0])   #获取图像背景，前景的分割点，即前景图像的位置
	n12=np.min(c[0])
	m13=np.max(c[1])
	n13=np.min(c[1])
	o=im1[n12:m12,n13:m13] #对各个序列图像进行背景去除
	return o

# read input (jpg格式文件)
#img = cv2.imread('2.jpg',1) #读取jpg格式文件

#read input (dicom格式文件)
im1=pydicom.dcmread('f:/50.dcm')  ##路径位置，可修改
im2=im1.pixel_array
n1=np.min(im2)
m1=np.max(im2)
im2=(im2-n1)/(m1-n1)
im2=im2*255.0
cv2.imwrite('1.jpg',im2)  ##存储位置， 可修改
img=cv2.imread('1.jpg',1) ##读取位置，可修改

img_blank=img.copy() #建立一个空图像便于后续显示颅骨区域
img_blank=np.zeros(shape=(img.shape[0],img.shape[1],img.shape[2]))


# convert to gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# threshold
t=iterative_threshold(gray)
thresh = cv2.threshold(gray, t , 255, cv2.THRESH_BINARY)[1]
cv2.imwrite('binary_img.jpg',thresh) #保存二值化图像，路径可修改

# find largest contour
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
big_contour = max(contours, key=cv2.contourArea)

draw_contours=cv2.drawContours(img_blank,big_contour,-1,(0,0,255),thickness=5) #绘制图像外轮廓
cv2.imwrite('contour.jpg',draw_contours) ##保存图像的外轮廓，路径可修改
cv2.imshow('contour',draw_contours)

# fit contour to ellipse and get ellipse center, minor and major diameters and angle in degree 
ellipse = cv2.fitEllipse(big_contour)
(xc,yc),(d1,d2),angle = ellipse
print(xc,yc,d1,d1,angle)

if angle<90:
	if angle<=5:
		angle_rotate=angle
	else:
		angle_rotate=5
	# height,width=img.shape[0:2]
	# retval=cv2.getRotationMatrix2D((xc,yc),angle,scale=1) #旋转矩阵
	# rotate_img=cv2.warpAffine(img,retval,(width,height))
else:
	if abs(angle-180)<=5:
		angle_rotate=angle-180
	else:
		angle_rotate=-5
height,width=img.shape[0:2]
retval=cv2.getRotationMatrix2D((xc,yc),angle_rotate,scale=1) #旋转矩阵
rotate_img=cv2.warpAffine(img,retval,(width,height))
cv2.imwrite('rotate_img.jpg',rotate_img)

# draw ellipse
result = img.copy()
cv2.ellipse(result, ellipse, (0, 255, 0), 3)

# draw circle at center
xc, yc = ellipse[0]
cv2.circle(result, (int(xc),int(yc)), 10, (255, 255, 255), -1)

# draw vertical line
# compute major radius
rmajor = max(d1,d2)/2
if angle > 90:
    angle = angle - 90
else:
    angle = angle + 90
print(angle)
xtop = xc + math.cos(math.radians(angle))*rmajor
ytop = yc + math.sin(math.radians(angle))*rmajor
# print(xtop,ytop)
xbot = xc + math.cos(math.radians(angle+180))*rmajor
ybot = yc + math.sin(math.radians(angle+180))*rmajor
cv2.line(result, (int(xtop),int(ytop)), (int(xbot),int(ybot)), (0, 0, 255), 2,cv2.LINE_AA)
cv2.circle(result, (int(xtop),int(ytop)), 10, (255, 0, 0), 3)
cv2.circle(result, (int(xbot),int(ybot)), 10, (255, 0, 0), 3)

cv2.imshow('input',img)
cv2.imwrite("labrador_ellipse.jpg", result)

cv2.imshow("labrador_thresh", thresh)
cv2.imshow("labrador_ellipse", result)
cv2.imshow('rotate',rotate_img)
cv2.waitKey(0)
cv2.destroyAllWindows()