#
# Adapted from https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767
#


from os import path
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import math

def sanity_check(data):
  for n in data:
    for t in n:
      for h in t:
        if math.isnan(h):
          print("nan detected")

## load data
data  = np.load("data.npy") 
labels= np.load("labels.npy")

## cross validation prep
data_train, data_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

## hyperparams
n_epochs = 100
batch_size = 4
num_classes = 2
learning_rate = 0.01
epsilon = 10**-8
n_data = len(data_train)		#n
n_seq = data_train.shape[1]		#t: time steps
n_syscall = data_train.shape[2]		#h: system call and args per time step
l_a = 0.001 # lambda attention model

## model params
initializer = tf.random_uniform_initializer(0,1)

batch_x = tf.placeholder(name="batch_x", shape=(batch_size, n_seq, n_syscall), dtype=tf.float32)
batch_y = tf.placeholder(name="batch_y", shape=(batch_size), dtype=tf.int32)
h0 = tf.placeholder(name="h0", shape=(batch_size, n_syscall), dtype=tf.float32) 
#attention_vector = tf.placeholder(name="attention_vector", shape=(n_seq), dtype=tf.float32)

W_hh = tf.get_variable("W_hh", shape=[n_syscall, n_syscall], initializer=initializer, dtype=tf.float32)
W_ih = tf.get_variable("W_ih", shape=[n_syscall, n_syscall], initializer=initializer, dtype=tf.float32)
W_ho = tf.get_variable("W_ho", shape=[n_syscall, num_classes], initializer=initializer, dtype=tf.float32)
b_o  = tf.get_variable("b_o",  shape=[batch_size,num_classes], initializer=initializer, dtype=tf.float32) 

W_a = tf.get_variable("W_a", shape=[n_syscall, n_syscall], initializer=initializer, dtype=tf.float32)


## model
layers_h = []
attention_vector = np.zeros((n_seq), dtype=np.float32)
h_prev = h0
for i in range(n_seq):
  x_t = batch_x[:,i,:] #nxh

  h_t = tf.tanh( tf.matmul(h_prev, W_hh) + tf.matmul(x_t, W_ih) )
  layers_h.append(h_t)
  h_prev = h_t

  #attention model steps
  a_t = tf.norm(tf.matmul(h_t, W_a))
  g_t = a_t * x_t
  ha_t = tf.tanh( tf.matmul(h_prev, W_hh) + g_t )

h_max = h_t

logits_series = tf.matmul(h_t, W_ho) + b_o + epsilon
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=batch_y, logits=logits_series) + l_a * tf.reduce_sum(attention_vector)
total_loss = tf.reduce_mean(losses)
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

## forward run
for epoch_idx in range(n_epochs):
  ## run epoch
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    h_init = np.zeros((batch_size, n_syscall), dtype=np.float32)

    for batch_pos in range(0, n_data, batch_size):
      data_batch = data_train[batch_pos:batch_pos + batch_size]
      labels_batch = y_train[batch_pos:batch_pos + batch_size].flatten()
      sanity_check(data_batch) # checks for nan values

      _total_loss, _train_step, h_init = sess.run([total_loss, train_step, h_t], feed_dict={batch_x:data_batch, batch_y:labels_batch, h0:h_init})
      print("Epoch: {}, Batch: {}, Loss: {}".format(epoch_idx, batch_pos // batch_size, _total_loss))

    G_writer = tf.summary.FileWriter('arsany/graph', sess.graph)

