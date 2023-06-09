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

LOG_DIR = 'E:/TensorBoard'
#SUMMARY_DIR = "D:/TensorBoard"
SPRITE_FILE = 'mnist_sprite.jpg'
META_FIEL = "mnist_meta.tsv"

def create_sprite_image(images):
    #"Returns a sprite image consisting of images passed as argument. Images should be count x width x height"
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
#    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
#    边长为10个图片
    n_plots = 10  
    print("n_plots:",n_plots)
    spriteimage = np.ones((img_h * n_plots ,img_w * n_plots ))

    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                spriteimage[i * img_h:(i + 1) * img_h,
                            j * img_w:(j + 1) * img_w] = this_img

    return spriteimage

mnist = input_data.read_data_sets("../../datasets/MNIST_data", one_hot=False)

#to_visualise = 1 - np.reshape(mnist.test.images[0:100],(-1,28,28))
to_visualise = 1 - np.reshape(mnist.test.images[0:100],(-1,28,28))
sprite_image = create_sprite_image(to_visualise)

path_for_mnist_sprites = os.path.join(LOG_DIR, SPRITE_FILE)
plt.imsave(path_for_mnist_sprites,sprite_image,cmap='gray')
plt.imshow(sprite_image,cmap='gray')

path_for_mnist_metadata = os.path.join(LOG_DIR, META_FIEL)

with open(path_for_mnist_metadata,'w') as f:
    f.write("E:/TensorBoard")
    for index,label in enumerate(mnist.test.labels[0:100]):
#    for index,label in enumerate(mnist.test.labels):
        f.write("%d\t%d\n" % (index,label))