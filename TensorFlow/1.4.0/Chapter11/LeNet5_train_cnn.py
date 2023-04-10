# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:39:22 2019

@author: wyhhhhhh
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import LeNet5_infernece_cnn
#import os
import numpy as np

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 6000
MOVING_AVERAGE_DECAY = 0.99

def train(mnist):
    # 定义输出为4维矩阵的placeholder
    with tf.name_scope('input_scope'):
        x = tf.placeholder(tf.float32, [
                BATCH_SIZE,
                LeNet5_infernece_cnn.IMAGE_SIZE,
                LeNet5_infernece_cnn.IMAGE_SIZE,
                LeNet5_infernece_cnn.NUM_CHANNELS],
                name='x-input')
        y_ = tf.placeholder(tf.float32, [None, LeNet5_infernece_cnn.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    
    with tf.name_scope('output_scope'):
        y = LeNet5_infernece_cnn.inference(x,False,regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数、学习率、滑动平均操作以及训练过程。
    with tf.name_scope("moving_average"):
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())  
#    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
#    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    with tf.name_scope('cross_entropy_mean'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
    with tf.name_scope('loss'):
         loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    with tf.name_scope('train_step'):
        learning_rate = tf.train.exponential_decay(
                LEARNING_RATE_BASE,
                global_step,
                mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY,staircase=True)
    
#        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)
        with tf.control_dependencies([train_step, variables_averages_op]):
            train_op = tf.no_op(name='train')
        
#    writer = tf.summary.FileWriter("E:/TensorBoard", tf.get_default_graph())
    # 初始化TensorFlow持久化类。\n",
#    saver = tf.train.Saver()
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter("E:/TensorBoard", sess.graph)
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            
            reshaped_xs = np.reshape(xs, (
                BATCH_SIZE,
                LeNet5_infernece_cnn.IMAGE_SIZE,
                LeNet5_infernece_cnn.IMAGE_SIZE,
                LeNet5_infernece_cnn.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})
#            summary_writer.add_summary(summary, i)

            if i == 0:
                print("After %d training step(s), loss on training batch is %g." % (i+1, loss_value))
            if i % 500 == 0:
                 # 配置运行时需要记录的信息。
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                # 运行时记录运行信息的proto。
                run_metadata = tf.RunMetadata()
                _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys}, options=run_options, run_metadata=run_metadata)
#                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                summary_writer.add_run_metadata(run_metadata=run_metadata, tag=("tag%d" % (i+1)), global_step=i)
                print("After %d training step(s), loss on training batch is %g." % (i+1, loss_value))
#    writer.close()
    summary_writer.close()
                    
def main(argv=None):
    mnist = input_data.read_data_sets("../../datasets/MNIST_data", one_hot=True)
    train(mnist)

#if __name__ == '__main__':
#    main()
    

if __name__ == '__main__':
    tf.app.run()