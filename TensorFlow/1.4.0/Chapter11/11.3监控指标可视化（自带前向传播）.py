# -*- coding: utf-8 -*-
"""
Created on Mon May 13 11:10:41 2019

@author: wyhhhhhh
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#SUMMARY_DIR = "/path/to/log"
#SUMMARY_DIR = "log"
#SUMMARY_DIR = "D://TensorBoard//test"
SUMMARY_DIR = "E:/TensorBoard"

BATCH_SIZE = 100
TRAIN_STEPS = 3000

def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        tf.summary.histogram(name, var)
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)
        
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1))
            variable_summaries(weights, layer_name + '/weights')
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
    mnist = input_data.read_data_sets("../../datasets/MNIST_data", one_hot=True)
#    mnist = input_data.read_data_sets("/path/to/MNIST_data\", one_hot=True)
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', image_shaped_input, 15)

    hidden1 = nn_layer(x, 784, 500, 'layer1')
    y = nn_layer(hidden1, 500, 10, 'layer2', act=tf.identity)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
        tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
    
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()
    
    with tf.Session() as sess:
 
        summary_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
        tf.global_variables_initializer().run()

        for i in range(TRAIN_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)

            summary, _ = sess.run([merged, train_step], feed_dict={x: xs, y_: ys})

            summary_writer.add_summary(summary, i)
    
    summary_writer.close()
      
#if __name__ == '__main__':
#    tf.app.run()
    
if __name__ == '__main__':
    main()