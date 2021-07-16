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
from model import inference
os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True #allocate dynamically

x_train=np.load('training_data.npy')
y_train=np.load('training_label.npy')
x_val=np.load('test_data1.npy')
y_val=np.load('test_label1.npy')
x_test=np.load('test_data2.npy')
y_test=np.load('test_label2.npy')

#将所有的图片resize成100*100
w=96
h=96
c=1

#-----------------构建网络----------------------
#占位符，设置输入参数的大小和格式
x=tf.placeholder(tf.float32,shape=[None,w,h,c],name='x')
y_=tf.placeholder(tf.int32,shape=[None,],name='y_')



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

#设置正则化参数为0.0001
regularizer = tf.contrib.layers.l2_regularizer(0.0001)
 
#将上述构建网络结构引入
logits,relu13,relu123 = inference(x,False,regularizer)
im1=conv_image_visual(relu13,96,96,4,8,32)
tf.summary.image('h_conv1',im1)
im2=conv_image_visual(relu123,96,96,16,4,64)
tf.summary.image('h_conv2',im2)


 
#(小处理)将logits乘以1赋值给logits_eval，定义name，方便在后续调用模型时通过tensor名字调用输出tensor
b = tf.constant(value=1,dtype=tf.float32)
logits_eval = tf.multiply(logits,b,name='logits_eval') 
 
#设置损失函数，作为模型训练优化的参考标准，loss越小，模型越优
loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_)
loss1=tf.add_n(tf.get_collection('losses'))
loss2=loss+loss1
#设置整体学习率为α为0.001
# batch_size=16
# global_step=tf.Variable(0,trainable=False)
# decay_ste=label1.shape[0]/batch_size
# learning_rate_base=0.01
# decay_rate=0.99
# learning_rate=tf.train.exponential_decay(learning_rate_base,global_step,decay_ste,decay_rate)
# train_op=tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)

train_op=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss2)

#设置预测精度
correct_prediction = tf.equal(tf.cast(tf.argmax(logits,1),tf.int32), y_)    
acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

merged = tf.summary.merge_all()

#定义一个函数，按批次取数据
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]
 
 
#训练和测试数据，可将n_epoch设置更大一些
 
#迭代次数
n_epoch=30  
#每次迭代输入的图片数据                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
batch_size=32
saver=tf.train.Saver(max_to_keep=1)
sess=tf.Session(config=config)  
train_writer = tf.summary.FileWriter("log/",sess.graph) 
#初始化全局参数
sess.run(tf.global_variables_initializer())
max_acc=0
#开始迭代训练，调用的都是前面设置好的函数或变量
train_cost=[]
train_accuracy=[]
val_cost=[]
val_accuracy=[]
for epoch in range(n_epoch):
    start_time = time.time()
 
    #training
    train_loss, train_acc, n_batch = 0, 0, 0
    i=0
    for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
        i=i+1
        _,err,ac,summary=sess.run([train_op,loss,acc,merged], feed_dict={x: x_train_a, y_: y_train_a})
        train_loss += err; train_acc += ac; n_batch += 1
    train_writer.add_summary(summary,epoch)
    print("   train loss: %f" % (np.sum(train_loss)/ n_batch))
    print("   train acc: %f" % (np.sum(train_acc)/ n_batch))
    train_cost.append(np.sum(train_loss)/ n_batch)
    train_accuracy.append(np.sum(train_acc)/ n_batch)
    #validation
    val_loss, val_acc, n_batch = 0, 0, 0
    for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
        err, ac = sess.run([loss,acc], feed_dict={x: x_val_a, y_: y_val_a})
        val_loss += err; val_acc += ac; n_batch += 1
    print("   validation loss: %f" % (np.sum(val_loss)/ n_batch))
    print("   validation acc: %f" % (np.sum(val_acc)/ n_batch))
    val_cost.append(np.sum(val_loss)/ n_batch)
    val_accuracy.append(np.sum(val_acc)/ n_batch)
	
    test_loss, test_acc, n_batch = 0, 0, 0
    for x_test_a, y_test_a in minibatches(x_test, y_test, batch_size, shuffle=False):
        err, ac = sess.run([loss,acc], feed_dict={x: x_test_a, y_: y_test_a})
        test_loss += err; test_acc += ac; n_batch += 1
    print("   test loss: %f" % (np.sum(test_loss)/ n_batch))
    print("   test acc: %f" % (np.sum(test_acc)/ n_batch))
    if val_acc>=max_acc:
        max_acc=val_acc;
        saver.save(sess,'ckpt2/train.ckpt',global_step=epoch+1)

# fig1=plt.figure(1,figsize=(8,6))
# plt.plot(train_cost,'r',label='training loss')
# plt.plot(val_cost,'b',label='validation loss')
# plt.xlabel('Epochs',fontdict={'family':'Times New Roman','size':24})
# plt.ylabel('Loss',fontdict={'family':'Times New Roman','size':24})
# plt.xticks(fontproperties='Times New Roman', size=18)
# plt.yticks(fontproperties='Times New Roman', size=18)
# plt.legend(prop={'family':'Times New Roman','size':24})
# plt.show()

# # 绘制精度函数曲线
# fig2=plt.figure(1,figsize=(8,6))
# plt.plot(train_accuracy,'r*-',label='training accuracy')
# plt.plot(val_accuracy,'bo-',label='validation accuracy')
# plt.xlabel('Epochs',fontdict={'family':'Times New Roman','size':24})
# plt.ylabel('Accuracy',fontdict={'family':'Times New Roman','size':24})
# plt.xticks(fontproperties='Times New Roman', size=18)
# plt.yticks(fontproperties='Times New Roman', size=18)
# plt.legend(prop={'family':'Times New Roman','size':24})
# plt.show()
# # #保存模型及模型参数
# # #各层特征可视化

# fig3, ax3=plt.subplots(figsize=(2,2))
# ax3.imshow(np.reshape(x_train[11],(96,96)))
# plt.show()

# #第一个卷积层可视化
# input_image = x_train[11:13]
# conv1_16 = sess.run(relu13, feed_dict={x:input_image})     # [1, 28, 28 ,16] 
# conv1_transpose = sess.run(tf.transpose(conv1_16, [3, 0, 1, 2]))
# fig4,ax4 = plt.subplots(nrows=1, ncols=32, figsize = (32,1))
# for i in range(32):
    # ax4[i].imshow(conv1_transpose[i][0],cmap=plt.cm.gray)                      # tensor的切片[row, column]
	
# #第一个卷积层可视化
# input_image = x_test[11:13]
# conv1_16 = sess.run(relu13, feed_dict={x:input_image})     # [1, 28, 28 ,16] 
# conv1_transpose = sess.run(tf.transpose(conv1_16, [3, 0, 1, 2]))
# fig4,ax4 = plt.subplots(nrows=1, ncols=32, figsize = (32,1))
# plt.xticks([])
# plt.yticks([])
# plt.axis('off')
# for i in range(32):
    # ax4[i].set_xticks([])
    # ax4[i].set_yticks([])
    # ax4[i].imshow(conv1_transpose[i][0])	# tensor的切片[row, column]
# plt.xticks([])
# plt.yticks([])
# plt.axis('off')
# plt.show()
	
# #第二个卷积层可视化
# input_image = x_test[11:13]
# conv2_16 = sess.run(relu123, feed_dict={x:input_image})     # [1, 28, 28 ,16] 
# conv2_transpose = sess.run(tf.transpose(conv2_16, [3, 0, 1, 2]))
# fig4,ax4 = plt.subplots(nrows=1, ncols=12, figsize = (12,1))
# plt.xticks([])
# plt.yticks([])
# plt.axis('off')
# for i in range(12):
    # ax4[i].set_xticks([])
    # ax4[i].set_yticks([])
    # ax4[i].imshow(conv2_transpose[i][0])	# tensor的切片[row, column]
# plt.show()
	
# #第三个卷积层可视化
# input_image = x_test[11:13]
# conv3_16 = sess.run(relu13, feed_dict={x:input_image})     # [1, 28, 28 ,16] 
# conv3_transpose = sess.run(tf.transpose(conv3_16, [3, 0, 1, 2]))
# fig4,ax4 = plt.subplots(nrows=1, ncols=12, figsize = (12,1))
# plt.xticks([])
# plt.yticks([])
# plt.axis('off')
# for i in range(12):
    # plt.xticks([])
    # plt.yticks([])
    # plt.axis('off')
    # ax4[i].set_xticks([])
    # ax4[i].set_yticks([])
    # ax4[i].imshow(conv3_transpose[i][0])
    # plt.xticks([])
    # plt.yticks([])
    # plt.axis('off')	
	# # tensor的切片[row, column]
# plt.show()
	
# #第四个卷积层可视化
# input_image = x_train[11:13]
# conv4_16 = sess.run(relu14, feed_dict={x:input_image})     # [1, 28, 28 ,16] 
# conv4_transpose = sess.run(tf.transpose(conv4_16, [3, 0, 1, 2]))
# fig4,ax4 = plt.subplots(nrows=1, ncols=16, figsize = (16,1))
# plt.xticks([])
# plt.yticks([])
# plt.axis('off')
# for i in range(16):
    # ax4[i].imshow(conv4_transpose[i][0])                      # tensor的切片[row, column]
	
#最后一个卷积层可视化
# input_image = x_test[11:13]
# conv5_16 = sess.run(relu4, feed_dict={x:input_image})     # [1, 28, 28 ,16] 
# conv5_transpose = sess.run(tf.transpose(conv5_16, [3, 0, 1, 2]))
# fig4,ax4 = plt.subplots(nrows=1, ncols=16, figsize = (16,1))
# plt.xticks([])
# plt.yticks([])
# plt.axis('off')
# for i in range(16):
    # ax4[i].set_xticks([])
    # ax4[i].set_yticks([])
    # ax4[i].imshow(conv5_transpose[i][0])   
	# # tensor的切片[row, column]
# plt.show()

# plt.title('Conv1 16x100x100')
# plt.show()
# sess.close()