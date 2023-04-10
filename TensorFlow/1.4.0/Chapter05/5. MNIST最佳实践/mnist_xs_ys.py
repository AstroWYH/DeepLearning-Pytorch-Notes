# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 22:09:22 2019

@author: wyhhhhhh
"""

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from keras.utils import np_utils

#import mnist_inference
#import os

INPUT_DATA = '../../../datasets/flower_processed_data4.npy'
#INPUT_DATA = '../../../datasets/flower_processed_data200200.npy'
INPUT_DATA = '../../../wyh/data64643_300_20191216.npy'

CLASS = 10

def train(mnist):
    with tf.Session() as sess:
        xs, ys = mnist.train.next_batch(100)
        reshaped_xs = np.reshape(xs, (100, 28, 28,1))
        
        print("mnist.test.images的形状:",mnist.test.images.shape)
        images = 1-np.reshape(mnist.test.images,(-1,28,28))
        print("images的形状:",images.shape)
        print("images.shape[0]:",images.shape[0])
        print("images.shape[1]:",images.shape[1])
        print("images.shape[2]:",images.shape[2])
        
        print("xs的形状:",xs.shape)
        print("xs的类型:",type(xs))
        print("ys的形状:",ys.shape)
        print("ys的类型:",type(ys))
        
        print("mnist.validation.images的形状:",mnist.validation.images.shape)
        print("mnist.validation.labels的形状:",mnist.validation.labels.shape)
        print("mnist.validation.images的长度:",len(mnist.validation.images))
        print("mnist.validation.labels的长度:",len(mnist.validation.labels))
        
        print("reshaped_xs的形状:",reshaped_xs.shape)
        print("reshaped_xs的类型:",type(reshaped_xs))  

        print("mnist.test.labels的形状:",mnist.test.labels.shape)
        print("mnist.test.labels的类型:",type(mnist.test.labels))
        print("mnist.test.labels的索引:",mnist.test.labels[0:2])

        print("-------我是分割线-------")
#       _x意思是改变后的labels 
       
        processed_data = np.load(INPUT_DATA)
        training_images = processed_data[0]
        n_training_example = len(training_images)
        training_labels = processed_data[1]
        print("training_images的len:",len(training_images))
        print("training_images的类型:",type(training_images))  
        print("training_labels[0:2]:",training_labels[0:2])
        print("training_labels[0:6]:",training_labels[0:6])
        
        print("np.array(training_images)的shape:",np.array(training_images).shape)
        print("np.array(training_images)的type:",type(np.array(training_images)))
        print("np.array(training_labels)的shape:",np.array(training_labels).shape)
        print("np.array(training_labels)的type:",type(np.array(training_labels)))
        
        training_labels_x = np_utils.to_categorical(training_labels, CLASS)
        print("training_labels_x的shape:\n",training_labels_x.shape)
        print("training_labels_x的type:\n",type(training_labels_x))
        print("np.array(training_labels_x)的shape:",np.array(training_labels_x).shape)
        print("np.array(training_labels_x)的type:",type(np.array(training_labels_x)))
        print("training_labels_x[0:2]:",training_labels_x[0:2])
        print("training_labels_x[0:6]:",training_labels_x[0:6])
        
        print("training_images[0:2]:",training_images[0:2])
        print("np.array(training_images)[0:2]转换成一维数组:",np.array(training_images)[0:2][0].flatten())
        
        a=[]
        for i in range(2):
            print("i =",i)
            x=np.array(training_images)[0:2][i].flatten()
            a.append(x)
            b=np.array(a)
        print("np.array(training_images)[0:2]拼成二维数组:",a)
        print("np.array(a)即b的shape:",b.shape)
        
        print("training_images的len:",len(training_images))
        print("training_images的类型:",type(training_images))  
        print("np.array(training_images)的shape:",np.array(training_images).shape)  

        a=[]
        for i in range(len(training_images)):
#            print("i =",i)
            x=np.array(training_images)[i].flatten()
            a.append(x)
            b=np.array(a)
#        print("np.array(training_images)拼成二维数组:",b)
        print("np.array(a)即b的shape:",b.shape)
#        img_list = []
#        for x in range(np.array(training_images)[0:1].size[0]):
#            for y in range(np.array(training_images)[0:1].size[1]):
#                img_list.append(np.array(training_images)[0:1][x,y])             
#        print("np.array(training_images)[0:1]转换成一维数组:",img_list)

        print("-------我是第二条分割线-------")
        
        print("mnist.train.images的长度:",len(mnist.train.images))
        print("mnist.train.images的shape:",mnist.train.images.shape)
        print("mnist.train.images的type:",type(mnist.train.images))
        print("mnist.train.images[0:2]:",mnist.train.images[0:2])
        
        print("mnist.train.labels的长度:",len(mnist.train.labels))
        print("mnist.train.labels的shape:",mnist.train.labels.shape)
        print("mnist.train.labels的type:",type(mnist.train.labels))        
        print("mnist.train.labels[0:2]:",mnist.train.labels[0:2])
        
        print("-------我是第三条分割线-------")
#       _x意思是改变后的labels, _1是轴承图像数据
        
        processed_data_1 = np.load(INPUT_DATA_1)
        training_images_1 = processed_data_1[0]
        n_training_example_1 = len(training_images_1)
        training_labels_1 = processed_data_1[1]
        
        validation_images_1 = processed_data_1[2]
        validation_labels_1 = processed_data_1[3]
        testing_imgaes_1 = processed_data_1[4]
        testing_labels_1 = processed_data_1[5]
                
        print("training_images_1的len:",len(training_images_1))
        print("training_images_1的类型:",type(training_images_1)) 
        print("training_labels_1[0:2]:",training_labels_1[0:2])
        print("training_labels_1[0:6]:",training_labels_1[0:6])    
        
        print("np.array(training_images_1)的shape:",np.array(training_images_1).shape)
        print("np.array(training_images_1)的type:",type(np.array(training_images_1)))
        print("np.array(training_labels_1)的shape:",np.array(training_labels_1).shape)
        print("np.array(training_labels_1)的type:",type(np.array(training_labels_1)))
        
        print("np.array(validation_images_1)的shape:",np.array(validation_images_1).shape)
        print("np.array(validation_labels_1)的shape:",np.array(validation_labels_1).shape)
        print("np.array(testing_imgaes_1)的shape:",np.array(testing_imgaes_1).shape)
        print("np.array(testing_labels_1)的shape:",np.array(testing_labels_1).shape)
        
        training_labels_1_x = np_utils.to_categorical(training_labels_1, CLASS)
        print("training_labels_1_x的shape:\n",training_labels_1_x.shape)
        print("training_labels_1_x的type:\n",type(training_labels_1_x))
        print("np.array(training_labels_1_x)的shape:",np.array(training_labels_1_x).shape)
        print("np.array(training_labels_1_x)的type:",type(np.array(training_labels_1_x)))
        print("training_labels_1_x[0:2]:",training_labels_1_x[0:2])
        print("training_labels_1_x[0:6]:",training_labels_1_x[0:6])
        
        print("np.array(training_images_1)的shape:",np.array(training_images_1).shape)  
        print("【注】：其实这个training_images_1的shape后续还会调整")
        print("training_images_1[0:2]:",training_images_1[0:2])      
        
        sess.run()

    
def main(argv=None):
    mnist = input_data.read_data_sets("../../../datasets/MNIST_data", one_hot=True)
    #mnist = input_data.read_data_sets("D:\TensorFlowCode\201806github\datasets\MNIST_data", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()

