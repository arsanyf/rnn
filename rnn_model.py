#
# Adapted from https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767
#


from os import path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
import math
#import matplotlib.pyplot as plt

def sanity_check(data):
  for n in data:
    for t in n:
      for h in t:
        assert not math.isnan(h)

## load data
data  = np.load("data.npy") 
labels= np.load("labels.npy")

## cross validation prep
data_train, data_test, y_train, y_test = train_test_split(data, labels, test_size=5)

## hyperparams
n_epochs = 10
batch_size = 5
num_classes = 2
learning_rate = 0.01
epsilon = 10**-8
n_data = len(data_train)		#n
n_seq = data_train.shape[1]		#t: time steps
n_syscall = data_train.shape[2]		#h: system call and args per time step
lambda_a = 0.001 # lambda attention model

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

W_a = tf.get_variable("W_a", shape=[n_syscall, 1], initializer=initializer, dtype=tf.float32)
W_u = tf.get_variable("W_u", shape=[n_syscall, n_syscall], initializer=initializer, dtype=tf.float32)

## model
layers_h = []
layers_ha= []
attention_vector = [] #np.zeros((batch_size, n_seq), dtype=np.float32)
feature_cost = np.ones((batch_size, n_seq), dtype=np.float32)
h_prev = h0
input_filtered = []
###attention layer
for i in range(n_seq):
  x_t = batch_x[:,i,:] #nxh

  h_t = tf.tanh( tf.matmul(h_prev, W_hh) + tf.matmul(x_t, W_ih) )
  layers_h.append(h_t)
  h_prev = h_t

  a_t = tf.matmul(h_t, W_a) #nxh
  attention_vector.append(a_t)
  g_t = tf.multiply(a_t, x_t) # Hadamard Product (only when training)
  input_filtered.append(g_t)

  ha_t = tf.tanh( tf.matmul(h_t, W_hh) + tf.matmul(g_t, W_u) )
  layers_ha.append(ha_t)

h_max = tf.convert_to_tensor(ha_t, dtype=tf.float32)

### loss calculation
f_cost = lambda_a * tf.reduce_sum(tf.matmul(feature_cost, tf.reshape(tf.convert_to_tensor(attention_vector), [n_seq, batch_size] )))
logits_series = tf.matmul(h_max, W_ho) + b_o + f_cost + epsilon
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=batch_y, logits=logits_series)
# + lambda_a * tf.reduce_sum(tf.matmul(feature_cost, tf.reshape(attention_vector, [n_seq, batch_size])))
total_loss = tf.reduce_mean(losses)

### accuracy calculation
probs_x = tf.cast(tf.argmax(tf.nn.softmax(logits_series), 1), tf.float32)
compare = tf.cast(tf.equal(tf.cast(batch_y, tf.float32), probs_x), tf.float32)
accuracy = tf.div(tf.reduce_sum(compare), batch_size, "Accuracy")
#fpr, tpr, thresholds = roc_curve(batch_y, probs_x, pos_label=1)
#auc = auc(fpr, tpr)

### training
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)


## forward run
loss_list = []
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  for epoch_idx in range(n_epochs):
  ## run epoch
#    plt.ion()
#    plt.figure()
#    plt.show()

    h_init = np.zeros((batch_size, n_syscall), dtype=np.float32)
    acc = 0
    for batch_pos in range(0, n_data, batch_size):
      data_batch = data_train[batch_pos:batch_pos + batch_size]
      labels_batch = y_train[batch_pos:batch_pos + batch_size].flatten()
      sanity_check(data_batch) # checks for nan values

      _total_loss, _f_cost, _train_step, h_init = sess.run([total_loss, f_cost, train_step, h_t], feed_dict={batch_x:data_batch, batch_y:labels_batch, h0:h_init})
      print("Epoch: {}, Batch: {}, Loss: {}, Cost: {}".format(epoch_idx+1, batch_pos // batch_size + 1, _total_loss, _f_cost))
      loss_list.append(_total_loss)
      #print("probs_x:{}".format(probs_x.eval(feed_dict={batch_x:data_batch, batch_y:labels_batch, h0:h_init})))
      #print("labels :{}".format(labels_batch))

#      if batch_pos  == n_data - batch_size: # plot at the end of the epoch
#        plt.plot(loss_list)

    ## validation
  _, v_loss, v_probs_x, v_acc = sess.run([h_t, total_loss, probs_x, accuracy], feed_dict={batch_x:data_test, batch_y:y_test.flatten(), h0:h_init})

  fpr, tpr, thresholds = roc_curve(y_test.flatten(), v_probs_x, pos_label=1)
  v_auc = auc(fpr, tpr)
  print("Validation>> Loss:{}, Accuracy: {}%, FPR: {}, TPR: {}, AUC: {}".format(v_loss, v_acc*100, fpr, tpr, v_auc))

  #G_writer = tf.summary.FileWriter('arsany/graph', sess.graph)

