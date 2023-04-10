# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 19:10:42 2019

@author: wyhhhhhh
"""

import numpy as np
import tensorflow as tf
from keras.utils import np_utils

print(np.__version__)

CLASS=5
label1=tf.constant([0,1,1,1,2,2,2,3,3,3,3,4,4,4,4])
sess1=tf.Session()
print('label1:\n',sess1.run(label1))
print('label1:\n',label1)
label2 = tf.one_hot(label1,CLASS)
with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    sess.run(label2)

    print('after one_hot的label2：\n',sess.run(label2))


#下面的代码转换十分重要哦！
N_CLASSES = 5
label = [0,1,1,1,2,2,2,3,3,3,3,4,4,4,4]
train_label = np_utils.to_categorical(label, N_CLASSES)
print("原始label:\n",label)
print("转换后的label\n",train_label)
print("np.array(label)的shape:\n",np.array(label).shape)
print("train_label的shape:\n",train_label.shape)
print("train_label的type:\n",type(train_label))