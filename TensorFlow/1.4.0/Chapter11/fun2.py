#import tensorflow as tf

#hello = tf.constant('Hello,Tensorflow')
#sess = tf.Session()
#print(sess.run(hello))

#sess = tf.Session()
#a = tf.constant(10)
#b= tf.constant(12)
#print(sess.run(a+b))

#import tensorflow as tf
#a = tf.constant([1.0,2.0,3.0],name='input1')
#b = tf.Variable(tf.random_uniform([3]),name='input2')
#add = tf.add_n([a,b],name='addOP')
#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    writer = tf.summary.FileWriter("D://TensorBoard//test",sess.graph)
#    print(sess.run(add))
#writer.close()

#import tensorflow as tf
#input1 = tf.constant([1.0,2.0,3.0],name="input1")
#input2 = tf.Variable(tf.random_uniform([3]),name="input2")
#output = tf.add_n([input1,input2],name="add")
#
#writer = tf.summary.FileWriter("/path/to/log",tf.get_default_graph())
#writer.close()
#


#import tensorflow as tf
## 定义一个简单的计算图，实现向量加法的操作。
#input1 = tf.constant([1.0, 2.0, 3.0], name = 'input1')
#input2 = tf.Variable(tf.random_uniform([3]), name = 'input2')
#output = tf.add_n([input1, input2], name = 'add')
## 生成一个写日志的writer，并将当前的tensorflow计算图写入日志。
## tensorflow提供了多种写日志文件的API
##writer = tf.summary.FileWriter('C:/logfile', tf.get_default_graph())
#writer = tf.summary.FileWriter('D:/TensorBoard', tf.get_default_graph())
#writer.close()

import tensorflow as tf
# 定义一个简单的计算图，实现向量加法的操作。
with tf.name_scope("input1"): 
    input1 = tf.constant([1.0, 2.0, 3.0], name = 'input1')
with tf.name_scope("input2"):
    input2 = tf.Variable(tf.random_uniform([3]), name = 'input2')
output = tf.add_n([input1, input2], name = 'add')

# 生成一个写日志的writer，并将当前的tensorflow计算图写入日志。
# tensorflow提供了多种写日志文件的API
#writer = tf.summary.FileWriter('C:/logfile', tf.get_default_graph())
writer = tf.summary.FileWriter('E:/TensorBoard', tf.get_default_graph())
writer.close()

