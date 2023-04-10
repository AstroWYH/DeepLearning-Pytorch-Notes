# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 16:17:32 2019

@author: wyhhhhhh
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

#image_raw_data = tf.gfile.FastGFile("",'r').read()
image_raw_data = tf.gfile.FastGFile("../../../datasets/cat.jpg",'rb').read()

with tf.Session() as sess:

    img_data = tf.image.decode_jpeg(image_raw_data)
    
    print(img_data.eval())
    #print img_data.eval()
    img_data.set_shape([1797, 2673, 3])
    print(img_data.get_shape())
   
    plt.imshow(img_data.eval())
    plt.show()
    
    encoded_image = tf.image.encode_jpeg(img_data)
    with tf.gfile.GFile("../../../datasets/cat_output","wb") as f:
        f.write(encoded_image.eval())