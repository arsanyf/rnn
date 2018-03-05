from os import path
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf


## load data
data  = np.load("data.npy")[1:]  # ignore first element as it is empty 
labels= np.load("labels.npy")

## cross validation prep
data_train, data_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

## model params
n_epochs = 10
batch_size = 4
num_classes = 2
learning_rate = 0.1
n_data = len(data_train)		#n
n_seq = data_train.shape[1]		#t
n_syscall = data_train.shape[2]		#h

## model
initializer = tf.random_uniform_initializer(-1,1)

batch_x = tf.placeholder(name="batch_x", shape=(batch_size, n_seq, n_syscall), dtype=tf.float32)
batch_y = tf.placeholder(name="batch_y", shape=(batch_size), dtype=tf.int32)
#x_t  = tf.placeholder(name="x_t" , shape=(batch_size, n_syscall), dtype=tf.float32)

h0 = tf.placeholder(name="h0", shape=(batch_size, n_syscall), dtype=tf.float32) 

W_hh = tf.get_variable("W_hh", shape=[n_syscall, n_syscall], initializer=initializer, dtype=tf.float32)
W_ih = tf.get_variable("W_ih", shape=[n_syscall, n_syscall], initializer=initializer, dtype=tf.float32)

W_ho = tf.get_variable("W_ho", shape=[n_syscall, 1], initializer=initializer, dtype=tf.float32)
b_o  = tf.get_variable("b_o",  shape=[batch_size, 1], initializer=initializer, dtype=tf.float32) 


## fwd
print("fwd")
h_prev = h0
#layers_h = []
for i in range(n_seq):
  x_t = batch_x[:,i,:] #nxh

  h_t = tf.tanh( tf.matmul(h_prev, W_hh) + tf.matmul(x_t, W_ih) )
  #layers_h.append(h_t)
  h_prev = h_t

h_max = h_t
logits_series = tf.matmul(h_max, W_ho) + b_o
#preds = tf.nn.softmax(logits_series)

losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=batch_y, logits=logits_series)
total_loss = tf.reduce_mean(losses)

train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

for epoch_idx in range(n_epochs):
  ## run epoch
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
  
    h_init = np.zeros((batch_size, n_syscall))
    #loss_list = []
    for batch_pos in range(0, n_data, batch_size):
      data_batch = data_train[batch_pos:batch_pos + batch_size]
      labels_batch = y_train[batch_pos:batch_pos + batch_size].flatten()

      print("processing data {} with labels {}".format(data_batch.shape, labels_batch.shape))
      _total_loss, _train_step, h_init = sess.run([total_loss, train_step, h_t], feed_dict={h0:h_init, batch_x:data_batch, batch_y:labels_batch})
      #loss_list.append(_total_loss)
      print("Epoch: {}, Batch: {}, Loss: {}".format(epoch_idx, batch_pos // batch_size, _total_loss))
      print(">W_hh: {}, W_ih:{}, W_ho:{}, b_o: {}".format(W_hh.eval(), W_ih.eval(), W_ho.eval(), b_o.eval()))
    G_writer = tf.summary.FileWriter('arsany/graph', sess.graph)

