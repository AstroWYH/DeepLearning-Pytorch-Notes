# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 18:41:58 2019

@author: 86198
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 11:49:21 2019

@author: wyhhhhhh
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
#INPUT_DATA = '../../wyh/data28.npy'
#INPUT_DATA = '../../wyh/data200.npy'
#INPUT_DATA = '../../wyh/data5005003_20191118.npy'
INPUT_DATA = '../../wyh/data64643_100_20191220.npy'
#INPUT_DATA = '../../datasets/flower_processed_data200200.npy'

from keras.utils import np_utils

CLASS=10

LOG_DIR = 'E:/TensorBoard'
#SUMMARY_DIR = "D:/TensorBoard"
SPRITE_FILE = 'mnist_sprite.jpg'
META_FIEL = "mnist_meta.tsv"

processed_data = np.load(INPUT_DATA)
training_images = processed_data[0]
n_training_example = len(training_images)
training_labels = processed_data[1]

training_images_temp=[]
for i in range(len(training_images)):
#            print("i =",i)
    temp_1d=np.array(training_images)[i].flatten() #.flatten把training_images)[0]的3维图像数据转换成1维数组，然后循环training_images)[1],training_images)[2]...
    training_images_temp.append(temp_1d) 
    training_images_feed=np.array(training_images_temp) #training_labels_feed是#training_labels_temp的ndarray形式

training_labels_feed = np_utils.to_categorical(training_labels, CLASS)

def create_sprite_image(images):
    #"Returns a sprite image consisting of images passed as argument. Images should be count x width x height"
    
    if isinstance(images, list):
        images = np.array(images)
    img_h = int(images.shape[1])
    img_w = int(images.shape[2])
#    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    n_plots = 10
    print("n_plots:",n_plots)
#    n_plots = 10

    spriteimage = np.ones((img_h * n_plots ,img_w * n_plots ))

    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                spriteimage[i * img_h:(i + 1) * img_h,
                            j * img_w:(j + 1) * img_w] = this_img

    return spriteimage

#mnist = input_data.read_data_sets("../../datasets/MNIST_data", one_hot=False)

#to_visualise = 1 - np.reshape(mnist.test.images[0:100],(-1,28,28))
#to_visualise = 1 - np.reshape(mnist.test.images,(-1,28,28))
#to_visualise = 1 - np.reshape(training_images_temp[0:100],(-1,28,28))  
to_visualise = 1 - np.reshape(training_images_feed,(-1,64,64))  
sprite_image = create_sprite_image(to_visualise)

path_for_mnist_sprites = os.path.join(LOG_DIR, SPRITE_FILE)
#plt.imsave(path_for_mnist_sprites,sprite_image,cmap='gray')
#plt.imshow(sprite_image,cmap='gray')

plt.imsave(path_for_mnist_sprites,sprite_image,cmap='gray_r')
plt.imshow(sprite_image,cmap='gray_r')

path_for_mnist_metadata = os.path.join(LOG_DIR, META_FIEL)

with open(path_for_mnist_metadata,'w') as f:
    f.write("E:/TensorBoard")
    for index,label in enumerate(training_labels_feed):
#    for index,label in enumerate(mnist.test.labels):
        f.write("%d\t%d\n" % (index,label))
