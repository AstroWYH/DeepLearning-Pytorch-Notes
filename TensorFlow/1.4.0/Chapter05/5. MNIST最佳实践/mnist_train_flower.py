# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 15:48:28 2019

@author: wyhhhhhh
"""



import mnist_inference
import glob
import os.path
import tensorflow as tf
from tensorflow.python.platform import gfile
import tensorflow.contrib.slim as slim
#from tensorflow.examples.tutorials.mnist import input_data
#import LeNet5_infernece_pic
from keras.utils import np_utils
import os
import numpy as np

BATCH_SIZE = 64
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 10000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH="E:/MODEL"
MODEL_NAME="mnist_train_flower_model"

#INPUT_DATA = '../../../datasets/flower_processed_data200200.npy'
INPUT_DATA = '../../../wyh/data64643_300_20191216.npy'
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
    
#    validation_images = processed_data[2]
#    validation_labels = processed_data[3]
#
#    testing_images = processed_data[4]
#    testing_labels = processed_data[5]

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
    #    y_ = tf.placeholder(tf.int64, [None, mnist_inference.OUTPUT_NODE], name='y-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
    #    y_ = tf.placeholder(tf.int64, [None], name='y-input')
    print("呵呵呵")
    
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    print("嘿嘿嘿")
    y = mnist_inference.inference(x, regularizer)
    print("嚯嚯嚯")
    global_step = tf.Variable(0, trainable=False)
    print("哈哈哈")
    
    with tf.name_scope("moving_average"):
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
#    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
#    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=tf.argmax(tf.one_hot(labels,CLASS), 1))

    with tf.name_scope("loss_function"):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    with tf.name_scope("train_step"):
        learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
        #        mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY,
            n_training_example / BATCH_SIZE, LEARNING_RATE_DECAY,
            staircase=True)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        
        with tf.control_dependencies([train_step, variables_averages_op]):
            train_op = tf.no_op(name='train')
        
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
    writer = tf.summary.FileWriter("E:/TensorBoard", tf.get_default_graph())

#    saver = tf.train.Saver()
    with tf.Session() as sess:
        #tf.get_variable_scope().reuse_variables() 没什么用 这两句本来想用来不让他每次重启才行
        #tf.reset_default_graph()
        tf.global_variables_initializer().run()

        start = 0
        end = BATCH_SIZE 
#        这一段是使用11.2.2节点信息之前的注释
#        for i in range(TRAINING_STEPS):
#        for i in range(TRAINING_STEPS):
##            xs, ys = mnist.train.next_batch(BATCH_SIZE)
##            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
#            _, training_accuracy, loss_value, step = sess.run([train_op, evaluation_step, loss, global_step], feed_dict={x: training_images_feed[start:end],y_: training_labels_feed[start:end]})
#            if i % 100 == 0:
##                print("xs:",xs.shape)
##                print("ys:",ys.shape)
##                print("After %d,%d training step(s), loss on training batch is %g." % (step, i, loss_value))
#                print('Step %d: Training loss is %.1f Training accuracy = %.1f%%' % (step, loss_value, training_accuracy * 100.0)) #loss改为loss_value, i改为step
##                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
        for i in range(TRAINING_STEPS):
#            xs, ys = mnist.train.next_batch(BATCH_SIZE)
#            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            
            if i % 100 == 0:
                
                run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                _, training_accuracy, loss_value, step = sess.run([train_op, evaluation_step, loss, global_step], feed_dict={x: training_images_feed[start:end],y_: training_labels_feed[start:end]},
                options = run_options, run_metadata = run_metadata)
                
#                writer.add_run_metadata(run_metadata, 'step%03d' % i) 书上的写法
                writer.add_run_metadata(run_metadata=run_metadata, tag=("tag%d" % i), global_step=i)
                print('Step %d: Training loss is %.1f Training accuracy = %.1f%%' % (step, loss_value, training_accuracy * 100.0)) #loss改为loss_value, i改为step
            else:
                 _, training_accuracy, loss_value, step = sess.run([train_op, evaluation_step, loss, global_step], feed_dict={x: training_images_feed[start:end],y_: training_labels_feed[start:end]})
                

#def main(argv=None):
#    mnist = input_data.read_data_sets("../../../datasets/MNIST_data", one_hot=True)
#    train(mnist)

    writer.close()

if __name__ == '__main__':
    main()

