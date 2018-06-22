# a sample code for building LSTM model in Tensorflow
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


input_size = 4
max_length = 60
hidden_size=64
output_size = 4

# 1. construct the model 
# (1). input layer or embedding layer here
x = tf.placeholder(tf.float32, shape=[None, max_length, input_size], name='x')
seqlen = tf.placeholder(tf.int64, shape=[None], name='seqlen')

# (2). define LSTM model architecture
lstm_cell = rnn.BasicLSTMCell(hidden_size, forget_bias=1.0)

outputs, states = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=x, sequence_length=seqlen, dtype=tf.float32)


encoded_states = states[-1]

W = tf.get_variable(
        name='W',
        shape=[hidden_size, output_size],
        dtype=tf.float32, 
        initializer=tf.random_normal_initializer())
b = tf.get_variable(
        name='b',
        shape=[output_size],
        dtype=tf.float32, 
        initializer=tf.random_normal_initializer())

z = tf.matmul(encoded_states, W) + b
results = tf.sigmoid(z)

# (3). compute the output of the model here

# (4). compute loss and build the optimizer here
###########################
## cost computing and training components goes here
# e.g. 
# targets = tf.placeholder(tf.float32, shape=[None, input_size], name='targets')
# cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=z))
# optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)
###############################

# 2. training the model
init = tf.global_variables_initializer()

batch_size = 4
data_in = np.zeros((batch_size, max_length, input_size), dtype='float32')
data_in[0, :4, :] = np.random.rand(4, input_size)
data_in[1, :6, :] = np.random.rand(6, input_size)
data_in[2, :20, :] = np.random.rand(20, input_size)
data_in[3, :, :] = np.random.rand(60, input_size)
data_len = np.asarray([4, 6, 20, 60], dtype='int64')


with tf.Session() as sess:
    sess.run(init)
    #########################
    # training process goes here
    #########################
    res = sess.run(results, 
            feed_dict={
                x: data_in, 
                seqlen: data_len})

print(res)