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
batch_size = 5
num_classes = 2
learning_rate = 0.01
n_data = len(data_train)		#n
n_seq = data_train.shape[1]		#t
n_syscall = data_train.shape[2]		#h

## model
initializer = tf.random_uniform_initializer(-1,1)

batch_x = tf.placeholder(name="batch_x", shape=(batch_size, n_seq, n_syscall), dtype=tf.float32)
batch_y = tf.placeholder(name="batch_y", shape=(batch_size, 1), dtype=tf.int32)
#x_t  = tf.placeholder(name="x_t" , shape=(batch_size, n_syscall), dtype=tf.float32)

h0 = tf.placeholder(name="h0", shape=(batch_size, n_syscall)) 

W_hh = tf.get_variable("W_hh", shape=[n_syscall, n_syscall], initializer=initializer, dtype=tf.float32)
W_ih = tf.get_variable("W_ih", shape=[n_syscall, n_syscall], initializer=initializer, dtype=tf.float32)

W_ho = tf.get_variable("W_ho", shape=[n_syscall, n_syscall], initializer=initializer, dtype=tf.float32)
b_o  = tf.get_variable("b_o",  shape=[batch_size, 1], initializer=initializer, dtype=tf.float32) 


## fwd
print("fwd")
h_prev = h0
layers_h = []
for i in range(n_seq)
  x_t = data_batch[:,i,:] #nxh

  h_t = tf.tanh( tf.matmul(h_prev, W_hh) + tf.matmul(x_t, W_ih) )
  layers_h.append(h_t)
  h_prev = h_t

logits_series = [tf.matmul(h, W_ho) + b_o for h in layers_h]
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]

losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels) for logits, labels in zip(logits_series,y_batch)]
total_loss = tf.reduce_mean(losses)

train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)

## run batch
with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())

  h_init = np.zeros((batch_size, n_syscall))
  for batch_pos in range(n_data):
    data_batch = data_train[batch_pos:batch_pos + batch_size]
    y_batch = y_train[batch_pos:batch_pos + batch_size]

    _total_loss, _train_step, _current_state, _predictions_series = sess.run([total_loss, train_step, current_state, predictions_series], feed_dict={h0=h_init,batch_x=data_batch, y=y_batch})
    loss_list.append(_total_loss)
    print("Loss:{}".format(_total_loss))
  

