# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 16:46:44 2019

@author: wyhhhhhh
"""


import mnist_inference_fcdnn
import glob
import os
import os.path
import tensorflow as tf
from tensorflow.python.platform import gfile
import tensorflow.contrib.slim as slim
#from tensorflow.examples.tutorials.mnist import input_data
#import LeNet5_infernece_pic
from keras.utils import np_utils
import numpy as np

from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 64
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 10000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH="E:/MODEL"
MODEL_NAME="mnist_train_flower_model"

LOG_DIR = 'E:/TensorBoard'
SPRITE_FILE = 'mnist_sprite.jpg'
META_FIEL = "mnist_meta.tsv"
TENSOR_NAME = "FINAL_LOGIT"

INPUT_DATA = '../../wyh/data64643_300_20191216.npy'
CLASS = 10

#def train(mnist):
def main(argv=None):
    
    # 加载预处理好的数据。
    processed_data = np.load(INPUT_DATA)
    training_images = np.array(processed_data)[0]
#    training_images = processed_data[0]
    n_training_example = len(training_images)
    training_labels = processed_data[1]
    
    training_images_temp=[]
    for i in range(len(training_images)):
#            print("i =",i)
        temp_1d=np.array(training_images)[i].flatten() #.flatten把training_images)[0]的3维图像数据转换成1维数组，然后循环training_images)[1],training_images)[2]...
        training_images_temp.append(temp_1d) 
        training_images_feed=np.array(training_images_temp) #training_labels_feed是#training_labels_temp的ndarray形式
        
    print("np.array(training_labels)的shape:\n",np.array(training_labels).shape)
    print("training_images_feed的shape:\n",training_images_feed.shape)
    
    training_labels_feed = np_utils.to_categorical(training_labels, CLASS)  #相当于tf.one_hot的功能，且不需要变成Tensor张量
    print("training_labels的长度:\n",len(training_labels))
    print("np.array(training_labels)的shape:\n",np.array(training_labels).shape)
    print("training_labels_feed的shape:\n",training_labels_feed.shape)
    print("training_labels_feed的type:\n",type(training_labels_feed))

    #  输入数据的命名空间。
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, mnist_inference_fcdnn.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference_fcdnn.OUTPUT_NODE], name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = mnist_inference_fcdnn.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 处理滑动平均的命名空间。
    with tf.name_scope("moving_average"):
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # 计算损失函数的命名空间。
    with tf.name_scope("loss_function"):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
#     定义学习率、优化方法及每一轮执行训练的操作的命名空间。
    with tf.name_scope("train_step"):
        learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            n_training_example / BATCH_SIZE, LEARNING_RATE_DECAY,
            staircase=True)

        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

        with tf.control_dependencies([train_step, variables_averages_op]):
            train_op = tf.no_op(name='train')

    # 训练模型。
#    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        
        start = 0
        end = BATCH_SIZE 

        for i in range(TRAINING_STEPS):
            
             _, training_accuracy, loss_value, step = sess.run([train_op, evaluation_step, loss, global_step], feed_dict={x: training_images_feed[start:end],y_: training_labels_feed[start:end]})              
            
            if i % 100 == 0:
                
                print('Step %d: Training loss is %.1f Training accuracy = %.1f%%' % (step, loss_value, training_accuracy * 100.0)) #loss改为loss_value, i改为step
           
        
#        for i in range(TRAINING_STEPS):
#            xs, ys = mnist.train.next_batch(BATCH_SIZE)
#            _, loss_value,training_accuracy, step = sess.run([train_op, loss,accuracy, global_step], feed_dict={x: xs, y_: ys})
#
#            if i % 1000 == 0:
#                print("After %d training step(s), loss on training batch is %g, training_accuracy is %g." % (i, loss_value, training_accuracy))
##                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
##                saver.save(sess, os.path.join(LOG_DIR, "model"), TRAINING_STEPS)
##        final_result = sess.run(y, feed_dict={x: mnist.test.images[0:100]})
        final_result = sess.run(y, feed_dict={x: mnist.test.images})
        visualisation(final_result)
#        print("final_result的个数：",final_result.shape)
#   
#    return final_result


def visualisation(final_result):
    y = tf.Variable(final_result, name = TENSOR_NAME)
    summary_writer = tf.summary.FileWriter(LOG_DIR)

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = y.name

    # Specify where you find the metadata
    embedding.metadata_path = META_FIEL

    # Specify where you find the sprite (we will create this later)
    embedding.sprite.image_path = SPRITE_FILE
#    embedding.sprite.single_image_dim.extend([28,28])
    embedding.sprite.single_image_dim.extend([64,64])

    # Say that you want to visualise the embeddings
    projector.visualize_embeddings(summary_writer, config)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
#    saver.save(sess, os.path.join(LOG_DIR, "model"), 2000)
    saver.save(sess, os.path.join(LOG_DIR,"model"),TRAINING_STEPS)

    summary_writer.close()
    
#def main(argv=None): 
#    mnist = input_data.read_data_sets("../../datasets/MNIST_data", one_hot=True)
#    final_result = train(mnist)
#    visualisation(final_result)

if __name__ == '__main__':
    main()