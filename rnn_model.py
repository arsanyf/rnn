#
# Adapted from https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767
#


from os import path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
import math
import matplotlib.pyplot as plt

def sanity_check(data):
  for n in data:
    for t in n:
      for h in t:
        assert not math.isnan(h)

## load data
data  = np.load("data_win3.npy") 
labels= np.load("labels_win3.npy")

## cross validation prep
data_train, data_test, y_train, y_test = train_test_split(data, labels, test_size=11, shuffle=True)

## hyperparams
n_epochs = 50
batch_size = 11
num_classes = 2
learning_rate = 0.001
epsilon = 10**-8
n_data = len(data_train)		#n
n_seq = data_train.shape[1]		#t: time steps
n_syscall = data_train.shape[2]		#h: system call and args per time step

## model params
initializer = tf.random_uniform_initializer(0,1)

batch_x = tf.placeholder(name="batch_x", shape=(batch_size, n_seq, n_syscall), dtype=tf.float32)
batch_y = tf.placeholder(name="batch_y", shape=(batch_size), dtype=tf.int32)
h0 = tf.placeholder(name="h0", shape=(batch_size, n_syscall), dtype=tf.float32) 

W_hh = tf.get_variable("W_hh", shape=[n_syscall, n_syscall], initializer=initializer, dtype=tf.float32)
W_ih = tf.get_variable("W_ih", shape=[n_syscall, n_syscall], initializer=initializer, dtype=tf.float32)
W_ho = tf.get_variable("W_ho", shape=[n_syscall, num_classes], initializer=initializer, dtype=tf.float32)
b_o  = tf.get_variable("b_o",  shape=[batch_size,num_classes], initializer=initializer, dtype=tf.float32) 

## model
layers_h = []
h_prev = h0
h_t=None
for i in range(n_seq):
  x_t = batch_x[:,i,:] #nxh

  h_t = tf.sigmoid( tf.matmul(h_prev, W_hh) + tf.matmul(x_t, W_ih) )
  layers_h.append(h_t)
  h_prev = h_t

  z_t = tf.nn.softmax(tf.matmul(h_t, W_ho) + b_o)
  dzdx = tf.reduce_mean(tf.norm(tf.gradients(z_t, x_t)))

### maxpooling
hs = tf.transpose(tf.convert_to_tensor(layers_h, dtype=tf.float32))
h_max = tf.nn.max_pool(tf.reshape(hs, [n_syscall, batch_size, n_seq, 1]), [1, 1, n_seq, 1], [1, 1, n_seq, 1], "VALID")
h_max = tf.transpose(tf.reshape(h_max, [n_syscall, batch_size]))

### loss calculation
logits_series = tf.matmul(h_max, W_ho) + b_o + epsilon
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=batch_y, logits=logits_series)
total_loss = tf.reduce_mean(losses)

### accuracy calculation
probs_x = tf.cast(tf.argmax(tf.nn.softmax(logits_series), 1), tf.float32)
compare = tf.cast(tf.equal(tf.cast(batch_y, tf.float32), probs_x), tf.float32)
accuracy = tf.div(tf.reduce_sum(compare), batch_size, "Accuracy")

### training
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)


## forward run
loss_list = []
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  for epoch_idx in range(n_epochs):
  ## run epoch

    h_init = np.zeros((batch_size, n_syscall), dtype=np.float32)
    acc = 0
    for batch_pos in range(0, n_data, batch_size):
      data_batch = data_train[batch_pos:batch_pos + batch_size]
      labels_batch = y_train[batch_pos:batch_pos + batch_size].flatten()
      #sanity_check(data_batch) # checks for nan values

      _total_loss, _, _train_step, h_init, _dzdx = sess.run([total_loss, h_max, train_step, h_t, dzdx], feed_dict={batch_x:data_batch, batch_y:labels_batch, h0:h_init})
      print("Epoch: {:2d}, Batch: {:2d}, Loss: {:.2f}, dzdx={}".format(epoch_idx+1, batch_pos // batch_size + 1, _total_loss, _dzdx))
      loss_list.append(_total_loss)
      #print("h_max:{}".format(h_max.eval(feed_dict={batch_x:data_batch, batch_y:labels_batch, h0:h_init})))
      #print("probs_x:{}".format(probs_x.eval(feed_dict={batch_x:data_batch, batch_y:labels_batch, h0:h_init})))
      #print("labels :{}".format(labels_batch))
      #with open("hs.txt", "a") as f:
      #  f.write("Epoch: {}, Batch: {}\n{}\n".format(epoch_idx +1 , batch_pos // batch_size + 1, h_max)) #.eval(feed_dict={batch_x:data_batch, batch_y:labels_batch, h0:h_init})))
#      if batch_pos  == n_data - batch_size: # plot at the end of the epoch
#        plt.plot(loss_list)

    ## validation
  _, v_loss, v_probs_x, v_acc = sess.run([h_t, total_loss, probs_x, accuracy], feed_dict={batch_x:data_test, batch_y:y_test.flatten(), h0:h_init})

  fpr, tpr, thresholds = roc_curve(y_test.flatten(), v_probs_x, pos_label=1)
  v_auc = auc(fpr, tpr)
  plt.plot(fpr, tpr, label="AUC={:.2f}\nLoss={:.2f}\nAccuracy={:.2f}%".format(v_auc, v_loss, v_acc*100))
  plt.legend(loc="lower right")
  plt.savefig("plots/{}.png".format(v_auc))
  print("Validation>> Loss:{:.2f}, Accuracy: {:.2f}%, FPR: {}, TPR: {}, AUC: {:.2f}".format(v_loss, v_acc*100, fpr, tpr, v_auc))

  #G_writer = tf.summary.FileWriter('arsany/graph', sess.graph)

