# -*- coding: utf-8 -*-
"""
Created on Mon May 13 11:10:41 2019

@author: wyhhhhhh
"""

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

#SUMMARY_DIR = "/path/to/log"
#SUMMARY_DIR = "log"
#SUMMARY_DIR = "D://TensorBoard//test"
SUMMARY_DIR = "E:/TensorBoard"

BATCH_SIZE = 64
#LEARNING_RATE_BASE = 0.8
#LEARNING_RATE_BASE = 0.01
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 10000
#MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH="E:/MODEL"
MODEL_NAME="mnist_train_flower_model"
INPUT_DATA = '../../wyh/data64643_300_20191216.npy'
CLASS = 10

def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        tf.summary.histogram(name, var)
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)
        
def nn_layer(input_tensor, input_dim, output_dim, layer_name, regularizer, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1))
            variable_summaries(weights, layer_name + '/weights')
            if regularizer != None: tf.add_to_collection('losses', regularizer(weights))  #regularizer只跟weights有关
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.constant(0.0, shape=[output_dim]))
            variable_summaries(biases, layer_name + '/biases')
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram(layer_name + '/pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram(layer_name + '/activations', activations)
        return activations

def main():
#    mnist = input_data.read_data_sets("../../datasets/MNIST_data", one_hot=True)

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
    
    with tf.name_scope('input_scope'):
        x = tf.placeholder(tf.float32, [None, 12288], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
    
    with tf.name_scope('regularizer'):
        regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    
    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x, [-1, 64, 64, 1])
        tf.summary.image('input', image_shaped_input, 10)
    
#    hidden1 = nn_layer(x, 784, 500, 'hidden1')
#    hidden2 = nn_layer(hidden1, 500, 100, 'hidden2')
#    hidden3 = nn_layer(hidden2, 100, 50, 'hidden3')
    
    hidden1 = nn_layer(x, 12288, 50, 'hidden1', regularizer)
#    hidden2 = nn_layer(hidden1, 50, 30, 'hidden2')
#    hidden3 = nn_layer(hidden2, 30, 20, 'hidden3')
    
    with tf.name_scope('output_scope'):
        y = nn_layer(hidden1, 50, 10, 'output', regularizer, act=tf.identity)
    global_step = tf.Variable(0, trainable=False) #居然源代码没有这句

#   滑动平均，学习率衰减，正则化还没有加入
#    with tf.name_scope("moving_average"):
#        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
#        variables_averages_op = variable_averages.apply(tf.trainable_variables())   
    
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
        tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('loss_function'):
        loss = cross_entropy + tf.add_n(tf.get_collection('losses'))
#        loss = cross_entropy
        tf.summary.scalar('loss_function', loss) #m目前没有加入正则化
    
    with tf.name_scope('train_step'):
        learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            n_training_example / BATCH_SIZE,
            LEARNING_RATE_DECAY,
            staircase=True)
#        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step = global_step)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = global_step)
    
#        with tf.control_dependencies([train_step, variables_averages_op]):
#            train_op = tf.no_op(name='train')
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()
    
#    with tf.Session() as sess:
# 
#        summary_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
#        tf.global_variables_initializer().run()
#
#        for i in range(TRAINING_STEPS):
#            xs, ys = mnist.train.next_batch(BATCH_SIZE)
##            mnist.validation.next_batch(BATCH_SIZE)
#
#            loss_value,step,summary, _ = sess.run([loss, global_step, merged, train_step], feed_dict={x: xs, y_: ys})
##            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
#
#            summary_writer.add_summary(summary, i)
#            
##            accuracy_validation = sess.run(accuracy, feed_dict = {x: mnist.validation.images, y_: mnist.validation.labels})
##            accuracy_test = sess.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels})
#            
##            if i == 0:
##                print("After %d training step(s), loss on training batch is %g, validation accuracy is %g, test accuracy is %g." % (i+1, loss_value, accuracy_validation, accuracy_test))
##            if (i+1) % 500 == 0:
##                print("After %d training step(s), loss on training batch is %g, validation accuracy is %g, test accuracy is %g." % (i+1, loss_value, accuracy_validation, accuracy_test))
#    
#    
#            if i == 0:
#                print("After %d training step(s), loss on training batch is %g." % (i+1, loss_value))
#
#            if (i+1) % 500 == 0:
#                # 配置运行时需要记录的信息。
#                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#                # 运行时记录运行信息的proto。
#                run_metadata = tf.RunMetadata()
#                loss_value,step,summary, _ = sess.run([loss, global_step, merged, train_step], feed_dict={x: xs, y_: ys})
#                _, loss_value, step = sess.run([train_step, loss, global_step], feed_dict={x: xs, y_: ys},options=run_options, run_metadata=run_metadata)
#                summary_writer.add_run_metadata(run_metadata=run_metadata, tag=("tag%d" % (i+1)), global_step=i)
#                print("After %d training step(s), loss on training batch is %g." % (i+1, loss_value))

    with tf.Session() as sess:
        #tf.get_variable_scope().reuse_variables() 没什么用 这两句本来想用来不让他每次重启才行
        #tf.reset_default_graph()
        summary_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
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

                _, training_accuracy, loss_value, step, summary = sess.run([train_step, accuracy, loss, global_step, merged], feed_dict={x: training_images_feed[start:end],y_: training_labels_feed[start:end]},
                options = run_options, run_metadata = run_metadata)
                
#                writer.add_run_metadata(run_metadata, 'step%03d' % i) 书上的写法
                summary_writer.add_run_metadata(run_metadata=run_metadata, tag=("tag%d" % i), global_step=i)
                print('Step %d: Training loss is %.1f Training accuracy = %.1f%%' % (step, loss_value, training_accuracy * 100.0)) #loss改为loss_value, i改为step
            else:
                 _, training_accuracy, loss_value, step, summary = sess.run([train_step, accuracy, loss, global_step, merged], feed_dict={x: training_images_feed[start:end],y_: training_labels_feed[start:end]})
                 summary_writer.add_summary(summary, i)
                
    summary_writer.close()
      
#if __name__ == '__main__':
#    tf.app.run()
    
if __name__ == '__main__':
    main()