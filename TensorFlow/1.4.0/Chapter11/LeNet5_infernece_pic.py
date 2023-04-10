#import tensorflow as tf
#
##INPUT_NODE = 784
##OUTPUT_NODE = 5
#OUTPUT_NODE = 10
#
#IMAGE_SIZE = 600
##IMAGE_SIZE = 299
##IMAGE_SIZE = 28
#
##NUM_CHANNELS = 1（这里是1还是3目前很难下定论）
##NUM_LABELS = 5
#NUM_CHANNELS = 3
#NUM_LABELS = 10
#
#CONV1_DEEP = 8
#CONV1_SIZE = 3
#
#CONV2_DEEP = 16
#CONV2_SIZE = 3
#
#FC_SIZE = 100
#
#def inference(input_tensor, train, regularizer):
#    with tf.variable_scope('layer1-conv1'):
#        conv1_weights = tf.get_variable(
#            "weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
#            initializer=tf.truncated_normal_initializer(stddev=0.1))
#        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
#        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
#        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
#
#    with tf.name_scope("layer2-pool1"):
#        pool1 = tf.nn.max_pool(relu1, ksize = [1,2,2,1],strides=[1,2,2,1],padding="SAME")
#
#    with tf.variable_scope("layer3-conv2"):
#        conv2_weights = tf.get_variable(
#            "weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
#            initializer=tf.truncated_normal_initializer(stddev=0.1))
#        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
#        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
#        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
#
#    with tf.name_scope("layer4-pool2"):
#        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#        pool_shape = pool2.get_shape().as_list()
#        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
#        reshaped = tf.reshape(pool2, [pool_shape[0], nodes])
#
#    with tf.variable_scope('layer5-fc1'):
#        fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],
#                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
#        if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))
#        fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))
#
#        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
#        if train: fc1 = tf.nn.dropout(fc1, 0.5)
#
#    with tf.variable_scope('layer6-fc2'):
#        fc2_weights = tf.get_variable("weight", [FC_SIZE, NUM_LABELS],
#                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
#        if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
#        fc2_biases = tf.get_variable("bias", [NUM_LABELS], initializer=tf.constant_initializer(0.1))
#        logit = tf.matmul(fc1, fc2_weights) + fc2_biases
#
#    return logit

import tensorflow as tf

#INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 64
NUM_CHANNELS = 3
NUM_LABELS = 10

CONV1_DEEP = 4
CONV1_SIZE = 5

CONV2_DEEP = 8
CONV2_SIZE = 5

CONV3_DEEP = 16
CONV3_SIZE = 3

CONV4_DEEP = 32
CONV4_SIZE = 3

CONV5_DEEP = 64
CONV5_SIZE = 3

FC_SIZE = 100

def inference(input_tensor, train, regularizer):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable(
            "weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

#    with tf.name_scope("layer2-pool1"):
    with tf.variable_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize = [1,2,2,1],strides=[1,2,2,1],padding="VALID")
        
    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable(
            "weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.variable_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize = [1,2,2,1],strides=[1,2,2,1],padding="VALID")

    with tf.variable_scope("layer5-conv3"):
        conv3_weights = tf.get_variable(
            "weight", [CONV3_SIZE, CONV3_SIZE, CONV2_DEEP, CONV3_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable("bias", [CONV3_DEEP], initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))
        
    with tf.variable_scope("layer6-pool3"):
        pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    with tf.variable_scope("layer7-conv4"):
        conv4_weights = tf.get_variable(
            "weight", [CONV4_SIZE, CONV4_SIZE, CONV3_DEEP, CONV4_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv4_biases = tf.get_variable("bias", [CONV4_DEEP], initializer=tf.constant_initializer(0.0))
        conv4 = tf.nn.conv2d(pool3, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))
        
    with tf.variable_scope("layer8-pool4"):
        pool4 = tf.nn.max_pool(relu4, ksize = [1,2,2,1],strides=[1,2,2,1],padding="VALID")

    with tf.variable_scope("layer9-conv5"):
        conv5_weights = tf.get_variable(
            "weight", [CONV5_SIZE, CONV5_SIZE, CONV4_DEEP, CONV5_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv5_biases = tf.get_variable("bias", [CONV5_DEEP], initializer=tf.constant_initializer(0.0))
        conv5 = tf.nn.conv2d(pool4, conv5_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu5 = tf.nn.relu(tf.nn.bias_add(conv5, conv5_biases))

#    with tf.name_scope("layer4-pool2"):
    with tf.variable_scope("layer10-pool5"):
#        pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        pool5 = tf.nn.max_pool(relu5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        pool_shape = pool5.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshaped = tf.reshape(pool5, [pool_shape[0], nodes])

    with tf.variable_scope('layer11-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train: fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer12-fc2'):
        fc2_weights = tf.get_variable("weight", [FC_SIZE, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    return logit

