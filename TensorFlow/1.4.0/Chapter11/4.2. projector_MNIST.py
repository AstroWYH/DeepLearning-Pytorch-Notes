# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 16:46:44 2019

@author: wyhhhhhh
"""

import tensorflow as tf
import mnist_inference
import os



from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

LOG_DIR = 'E:/TensorBoard'
SPRITE_FILE = 'mnist_sprite.jpg'
META_FIEL = "mnist_meta.tsv"
TENSOR_NAME = "FINAL_LOGIT"

def train(mnist):
    #  输入数据的命名空间。
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = mnist_inference.inference(x, regularizer)
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
            mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY,
            staircase=True)

        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

        with tf.control_dependencies([train_step, variables_averages_op]):
            train_op = tf.no_op(name='train')

    # 训练模型。
#    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value,training_accuracy, step = sess.run([train_op, loss,accuracy, global_step], feed_dict={x: xs, y_: ys})

            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g, training_accuracy is %g." % (i, loss_value, training_accuracy))
#                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
#                saver.save(sess, os.path.join(LOG_DIR, "model"), TRAINING_STEPS)
#        final_result = sess.run(y, feed_dict={x: mnist.test.images[0:100]})
        final_result = sess.run(y, feed_dict={x: mnist.test.images})
#        print("final_result的个数：",final_result.shape)
#   
    return final_result

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
    embedding.sprite.single_image_dim.extend([100,100])

    # Say that you want to visualise the embeddings
    projector.visualize_embeddings(summary_writer, config)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
#    saver.save(sess, os.path.join(LOG_DIR, "model"), 2000)
    saver.save(sess, os.path.join(LOG_DIR,"model"),TRAINING_STEPS)

    summary_writer.close()
    
def main(argv=None): 
    mnist = input_data.read_data_sets("../../datasets/MNIST_data", one_hot=True)
    final_result = train(mnist)
    visualisation(final_result)

if __name__ == '__main__':
    main()