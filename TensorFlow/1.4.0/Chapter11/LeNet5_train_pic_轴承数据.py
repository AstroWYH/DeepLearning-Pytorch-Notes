# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:39:22 2019

@author: wyhhhhhh
"""
import glob
import os.path
import tensorflow as tf
from tensorflow.python.platform import gfile
import tensorflow.contrib.slim as slim
#from tensorflow.examples.tutorials.mnist import input_data
import LeNet5_infernece_pic
import os
import numpy as np
from keras.utils import np_utils

#INPUT_DATA = '../../datasets/flower_processed_data200200.npy'
INPUT_DATA = '../../wyh/data64643_300_20191216.npy'

#BATCH_SIZE = 100
BATCH_SIZE = 64
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 4600
#TRAINING_STEPS = 300
MOVING_AVERAGE_DECAY = 0.99

#BATCH = 32
#STEPS = 300
#LEARNING_RATE = 0.0001
#CLASS = 5(看分类的情况)
CLASS = 10

def main(argv=None): #def train(mnist):改为def main():
    # 定义输出为4维矩阵的placeholder
#    x = tf.placeholder(tf.float32, [
#            BATCH_SIZE,
#            LeNet5_infernece.IMAGE_SIZE,
#            LeNet5_infernece.IMAGE_SIZE,
#            LeNet5_infernece.NUM_CHANNELS],
#            name='x-input')
#    y_ = tf.placeholder(tf.float32, [None, LeNet5_infernece.OUTPUT_NODE], name='y-input')
    
    # 加载预处理好的数据。
    processed_data = np.load(INPUT_DATA)
    training_images = processed_data[0]
    n_training_example = len(training_images)
    training_labels = processed_data[1]
    
    validation_images = processed_data[2]
    validation_labels = processed_data[3]
    testing_images = processed_data[4]
    testing_labels = processed_data[5]

#   下面这段程序是training_images到底是(797,100,100,3)还是(797,30000)
#    training_images_temp=[]
#    for i in range(len(training_images)):
##            print("i =",i)
#        temp_1d=np.array(training_images)[i].flatten() #.flatten把training_images)[0]的3维图像数据转换成1维数组，然后循环training_images)[1],training_images)[2]...
#        training_images_temp.append(temp_1d) 
#        training_images_feed=np.array(training_images_temp) #training_labels_feed是#training_labels_temp的ndarray形式
        
    print("np.array(training_images)的shape:\n",np.array(training_images).shape)
    print("np.array(validation_images)的shape:\n",np.array(validation_images).shape)
    print("np.array(testing_labels)的shape:\n",np.array(testing_labels).shape)
#    print("np.array(training_images_feed)的shape:\n",np.array(training_images_feed).shape)
    
    training_labels_feed = np_utils.to_categorical(training_labels, CLASS)
    validation_labels_feed = np_utils.to_categorical(validation_labels, CLASS)
    testing_labels_feed = np_utils.to_categorical(testing_labels, CLASS)
    
    print("training_labels的长度:\n",len(training_labels))
    print("np.array(training_labels)的shape:\n",np.array(training_labels).shape)
    print("np.array(training_labels)的shape:\n",np.array(training_labels).shape)
    print("training_labels_feed的shape:\n",training_labels_feed.shape)
    print("training_labels_feed的type:\n",type(training_labels_feed))
    
    print("validation_labels_feed的shape:\n",validation_labels_feed.shape)
    print("testing_labels_feed的shape:\n",testing_labels_feed.shape)
    
#    validation_images = processed_data[2]
#    validation_labels = processed_data[3]
#
#    testing_images = processed_data[4]
#    testing_labels = processed_data[5]
    
#    print("%d training examples, %d validation examples and %d testing examples." % (n_training_example, len(validation_labels), len(testing_labels)))
    print("嚯嚯嚯嚯嚯嚯")

#    images = tf.placeholder(tf.float32, [None, 299, 299, 3], name='input_images')
#    labels = tf.placeholder(tf.int64, [None], name='labels')
    
#    x = tf.placeholder(tf.float32, [BATCH_SIZE, LeNet5_infernece_pic.IMAGE_SIZE, LeNet5_infernece_pic.IMAGE_SIZE, LeNet5_infernece_pic.NUM_CHANNELS], name='x-input')
    
    x = tf.placeholder(tf.float32, [BATCH_SIZE, LeNet5_infernece_pic.IMAGE_SIZE, LeNet5_infernece_pic.IMAGE_SIZE, LeNet5_infernece_pic.NUM_CHANNELS], name='x-input')    
#    images=tf.image.convert_image_dtype(images,tf.float32)
#    images = tf.placeholder(tf.float32, [None, 28, 28, 3], name='input_images')
#    labels = tf.placeholder(tf.float32, [None, LeNet5_infernece_pic.OUTPUT_NODE], name='labels')    #上面把数字换成了LeNet5_infernece_pic.IMAGE_SIZE等
#    labels = tf.placeholder(tf.int64, [None], name='labels')    #上面把数字换成了LeNet5_infernece_pic.IMAGE_SIZE等
    
    y_ = tf.placeholder(tf.float32, [None, LeNet5_infernece_pic.OUTPUT_NODE], name='y-input')    #上面把数字换成了LeNet5_infernece_pic.IMAGE_SIZE等
    
#    y_ = tf.placeholder(tf.int64, [None], name='y-input')    #上面把数字换成了LeNet5_infernece_pic.IMAGE_SIZE等
    
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
#    y = LeNet5_infernece.inference(x,False,regularizer)
    y = LeNet5_infernece_pic.inference(x,False,regularizer) #x改为images, y改为output
    global_step = tf.Variable(0, trainable=False)

    print("呵呵呵呵呵呵呵呵")
    # 定义损失函数、学习率、滑动平均操作以及训练过程。
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
#    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=tf.argmax(tf.one_hot(labels,CLASS), 1))
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)               #tf.one_hot(output,CLASS,1,0), y_改为tf.one_hot(labels,CLASS,1,0)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
#            mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY,staircase=True)
            n_training_example / BATCH_SIZE, LEARNING_RATE_DECAY,staircase=True)
    

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
#    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
    
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')
        
#    correct_prediction = tf.equal(tf.argmax(logits, 1), labels) 
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 初始化TensorFlow持久化类。\n",
#    saver = tf.train.Saver()
#    with tf.Session() as sess:
#        tf.global_variables_initializer().run()
#        for i in range(TRAINING_STEPS):
#            xs, ys = mnist.train.next_batch(BATCH_SIZE)
#            
#            reshaped_xs = np.reshape(xs, (
#                BATCH_SIZE,
#                LeNet5_infernece.IMAGE_SIZE,
#                LeNet5_infernece.IMAGE_SIZE,
#                LeNet5_infernece.NUM_CHANNELS))
#            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})
#
#            if i % 1000 == 0:
#                    print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
    
    print("哈哈哈哈哈哈哈哈哈哈哈哈哈")
    
    with tf.Session() as sess:
        # 初始化没有加载进来的变量。
        init = tf.global_variables_initializer()
        sess.run(init)

        # 加载谷歌已经训练好的模型。
        # print('Loading tuned variables from %s' % CKPT_FILE)
        # load_fn(sess)

        start = 0
        end = BATCH_SIZE #BATCH改为BATCH_SIZE
#        for i in range(TRAINING_STEPS): #STEPS改为TRAINING_STEPS
#            reshaped_xs = np.reshape(training_images_feed[start:end], (
#                BATCH_SIZE,
#                LeNet5_infernece_pic.IMAGE_SIZE,
#                LeNet5_infernece_pic.IMAGE_SIZE,
#                LeNet5_infernece_pic.NUM_CHANNELS))
#            _, training_accuracy, loss_value, step = sess.run([train_op, evaluation_step, loss, global_step], feed_dict={ #loss改为loss_value, total_loss改为loss,
#                    x: reshaped_xs,    #0,1,2                           #train_step改为train_op, 等号左边增加step, 右边增加global_step
#                    y_: training_labels_feed[start:end]})
        
        for i in range(TRAINING_STEPS): #STEPS改为TRAINING_STEPS
            _, training_accuracy, loss_value, step = sess.run([train_op, evaluation_step, loss, global_step], feed_dict={ #loss改为loss_value, total_loss改为loss,
                    x: training_images[start:end],    #0,1,2                           #train_step改为train_op, 等号左边增加step, 右边增加global_step
                    y_: training_labels_feed[start:end]})
            if i % 10 == 0:
##                print("training_images[start:end]:",training_images[start:end])
#                print("training_labels[start:end]:",training_labels[start:end])
##                print("training_images[start:end]的形状:",training_images[start:end].shape)
##                print("training_labels[start:end]的形状:",training_labels[start:end].shape)
#                print("training_images[start:end]的长度:",len(training_images[start:end]))
#                print("training_labels[start:end]的长度:",len(training_labels[start:end]))
#                print("training_images[start:end]的类型:",type(training_images[start:end]))
#                print("training_labels[start:end]的类型:",type(training_labels[start:end]))
                
                print('Step %d: Training loss is %.1f Training accuracy = %.1f%%' % (step, loss_value, training_accuracy * 100.0)) #loss改为loss_value, i改为step
#            if i % 30 == 0 or i + 1 == TRAINING_STEPS: #STEPS改为TRAINING_STEPS
##                saver.save(sess, TRAIN_FILE, global_step=i)
#
#                validation_accuracy = sess.run(evaluation_step, feed_dict={images: validation_images, labels: tf.one_hot(validation_labels,5)})
#                print('Step %d: Training loss is %.1f Validation accuracy = %.1f%%' % (step, loss_value, validation_accuracy * 100.0)) #loss改为loss_value, i改为step

#            if i % 300 == 0 and i >= 300:
##                for i in range(TRAINING_STEPS): #STEPS改为TRAINING_STEPS
#                    _, testing_accuracy, loss_value, step = sess.run([train_op, evaluation_step, loss, global_step], feed_dict={ #loss改为loss_value, total_loss改为loss,
#                            x: testing_images[0:16],    #0,1,2                           #train_step改为train_op, 等号左边增加step, 右边增加global_step
#                            y_: testing_labels_feed[0:16]})
##                    if i % 10 == 0:
#                    print('Step %d: Training loss is %.1f testing accuracy = %.1f%%' % (step, loss_value, testing_accuracy * 100.0)) #loss改为loss_value, i改为step



            start = end
            if start == n_training_example or start > n_training_example:
                start = 0

            end = start + BATCH_SIZE #BATCH改为BATCH_SIZE
            if end > n_training_example or end == n_training_example:
#                end = n_training_example
                start = 0
                end = BATCH_SIZE
                
#            if i % 1000 == 0:
#                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))

        # 在最后的测试数据上测试正确率。\n",
        # test_accuracy = sess.run(evaluation_step, feed_dict={\n",
            # images: testing_images, labels: testing_labels})\n",
        # print('Final test accuracy = %.1f%%' % (test_accuracy * 100))"
                   
#def main(argv=None):
#    mnist = input_data.read_data_sets("../../../datasets/MNIST_data", one_hot=True)
#    train(mnist)

if __name__ == '__main__':
    main()
    
#if __name__ == '__main__':
#    tf.app.run()