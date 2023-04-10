# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 20:47:06 2019

@author: wyhhhhhh
"""

import glob
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

# 原始输入数据的目录，这个目录下有5个子目录，每个子目录底下保存这属于该
# 类别的所有图片。
#INPUT_DATA = '../../datasets/flower_photos'
INPUT_DATA = '../../wyh/data100'
# 输出文件地址。我们将整理后的图片数据通过numpy的格式保存。
OUTPUT_FILE = '../../wyh/data64643_100_20191220.npy'
#OUTPUT_FILE = '../../datasets/flower20191118_150_150.npy'

# 测试数据和验证数据比例。
VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10

# 读取数据并将数据分割成训练数据、验证数据和测试数据。
def create_image_lists(sess, testing_percentage, validation_percentage):
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    is_root_dir = True

    # 初始化各个数据集。
    training_images = []
    training_labels = []
    testing_images = []
    testing_labels = []
    validation_images = []
    validation_labels = []
    current_label = 0

    # 读取所有的子目录。
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        # 获取一个子目录中所有的图片文件。\n",
#        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        extensions = ['jpg', 'jpeg']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
#            print(extension)
#            print(file_list)
        if not file_list: continue
    
        print("processing:", dir_name)
#        print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
        
        i = 0
        # 处理图片数据。
        for file_name in file_list:
            i += 1
            # 读取并解析图片，将图片转化为299*299以方便inception-v3模型来处理。
            image_raw_data = gfile.FastGFile(file_name, 'rb').read()
            image = tf.image.decode_jpeg(image_raw_data)
            if image.dtype != tf.float32:
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image = tf.image.resize_images(image, [64, 64])
            image_value = sess.run(image)
            #print("image_value",image_value)    
            
            # 随机划分数据聚。
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(image_value)
                validation_labels.append(current_label)

#                print("validation_images", validation_images)
            elif chance < (testing_percentage + validation_percentage):
                testing_images.append(image_value)
                testing_labels.append(current_label)
            else:
                training_images.append(image_value)
                training_labels.append(current_label)
#                print("training_images", training_images)
#            if i % 5 == 0:
            if i % 100 == 0:          
                print(i, "images processed.")
#            print(file_name)
#            print(file_list)
        current_label += 1

    # 将训练数据随机打乱以获得更好的训练效果。
    state = np.random.get_state()
    np.random.shuffle(training_images)
    np.random.set_state(state)
    np.random.shuffle(training_labels)

#    print("validation_images:", len(validation_images))
#    print("validation_images转换成ndarray:", np.array(validation_images).shape)
#    print("validation_labels:", len(validation_labels))
#    print("validation_labels",validation_labels)
#    print("validation_labels的type:",type(validation_labels))
#    print("validation_labels经过tf.one_hot转换的type:",type(tf.one_hot(validation_labels,5)))
#    print("validation_labels",tf.one_hot(validation_labels,5))
#    print("validation_labels",tf.one_hot(validation_labels,5).shape)
    
    return np.asarray([training_images, training_labels,
                       validation_images, validation_labels,
                       testing_images, testing_labels])

def main():
    with tf.Session() as sess:
        processed_data = create_image_lists(sess, TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
        # 通过numpy格式保存处理后的数据。
        np.save(OUTPUT_FILE, processed_data)
        
if __name__ == '__main__':
    main()