from os import path
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

class RNNModel():
  def __init__(self, data, labels):
    ## config
    self.n=data_train.shape[0] # number of malware/benign samples
    self.t=data.shape[1] # number of syscalls per sample
    self.h=data.shape[2] # features collected per syscall (syscall + params)

    self.data = data
#    self.labels = labels

    self.init_params()

  def init_params(self):
    initializer = tf.random_uniform_initializer(-1,1)

    #self.learning_rate = 0.01

    self.W_hh = tf.get_variable("W_hh", shape=[self.h,self.h],initializer=initializer,dtype=tf.float32) #tf.random_normal((self.h,self.h), 0, 1.0, tf.float64)
    self.W_ih = tf.get_variable("W_ih", shape=[self.h, self.h],initializer=initializer,dtype=tf.float32) #tf.random_normal((self.h,self.n), 0, 1.0, tf.float64)
    #self.b_h  = tf.get_variable("b_h", shape=[self.n, 1],initializer=initializer,dtype=tf.float32) #tf.random_normal((self.h,self.h), 0, 1.0, tf.float64)
    #self.b_h_tiled = tf.convert_to_tensor( np.tile(self.b_h.eval(), self.h), dtype=tf.float32 )
    #self.x_t  = tf.placeholder(name="x_t" , shape=(self.n,self.h),dtype=tf.float32)
    self.h_prev=tf.get_variable("h_prev", shape=[self.n,self.h],initializer=initializer,dtype=tf.float32) #= tf.random_normal((self.h,self.h), 0, 1.0, tf.float64)
    self.h_t  = tf.get_variable("h_t", shape=[self.n,self.h],initializer=initializer,dtype=tf.float32) #tf.random_normal((self.h,self.h), 0, 1.0, tf.float64)
    self.W_ho = tf.get_variable("W_ho", shape=[self.h,self.h],initializer=initializer,dtype=tf.float32) #tf.random_normal((self.h,self.h), 0, 1.0, tf.float64)
    self.b_o  = tf.get_variable("b_o", shape=[self.n, 1],initializer=initializer,dtype=tf.float32) #tf.random_normal((self.h,self.h), 0, 1.0, tf.float64)
    self.z_t  = tf.get_variable("z_t", shape=[self.n, self.h], dtype=tf.float32) #tf.nn.softmax(tf.matmul(self.W_ho, self.h_t) + self.b_o)
    self.z_t_max = tf.get_variable("z_t_max", shape=[self.n, self.h], dtype=tf.float32)
    self.loss = tf.get_variable("loss",shape=[1], dtype=tf.float32) 
    self.layers_h = tf.get_variable("layers_h",shape=[self.n,self.h,self.t],dtype=tf.float32) #[]
    self.layers_z = tf.get_variable("layers_z",shape=[self.n,self.h,self.t],dtype=tf.float32) #[]
    self.W_pred = tf.get_variable("W_pred",shape=[self.h, 1],initializer=initializer,dtype=tf.float32) # linear layer
    self.b_pred = tf.get_variable("b_pred",shape=[self.n, 1],initializer=initializer,dtype=tf.float32)
    self.preds  = tf.get_variable("preds",shape=[self.n, 1],dtype=tf.float32)
    self.labels = tf.get_variable("labels",shape=[self.n,1],dtype=tf.float32)
    self.labels = labels

  def RU(self, time_step):
    """
    Returns h_t
    """
    #b_h_tiled = tf.convert_to_tensor( np.tile(self.b_h.eval(), self.h), dtype=tf.float32 )
    return tf.tanh( tf.matmul(self.h_prev, self.W_hh) + tf.matmul(time_step, self.W_ih) ) #+ self.b_h_tiled )

#  def loss_fn(self, prediction):
#    """
#    Returns loss function per time step = y_t * log(z_t)
#    """
#    tmp_labels = tf.convert_to_tensor( np.tile(self.labels, self.h), dtype=tf.float32 )
#    return tf.matmul( tmp_labels, tf.log(prediction) )


  def fwd_prop(self):
    """
    Perform forward propapagation as follows:
    for each time step:
    1) Calculate h_t = tanh(W_hh * h_prev + W_ih * x_t + b_h)
    2) Calculate z_t = softmax(W_ho * h_t + b_o)
    3) Calculate z_t_max
    4) Calculate predictions = sotmax(W_pred * z_t_max)
    5) Calculate loss using softmax cross entropy
    """
    print("fwd...")
    self.loss = 0.0
    tmp_layers_h = []
    tmp_layers_z = []
    for i in range(self.t):
      time_step = tf.convert_to_tensor(self.data[:,i,:], dtype=tf.float32) # n x h
      self.h_prev = self.h_t # n x h
      self.h_t = self.RU(time_step) # n x h 
      tmp_layers_h.append(self.h_t)
      
      self.z_t = tf.nn.softmax(tf.matmul(self.h_t, self.W_ho) + tf.convert_to_tensor( np.tile(self.b_o.eval(), self.h), dtype=tf.float32 ) ) # n x h
      tmp_layers_z.append(self.z_t)

      #self.loss += self.loss_fn(self.z_t)
    self.layers_h = tf.stack(tmp_layers_h)  # https://stackoverflow.com/a/37706972
    self.layers_z = tf.stack(tmp_layers_z)
    self.z_t_max = self.z_t
    self.preds = tf.nn.softmax( tf.matmul(self.z_t_max, self.W_pred) + self.b_pred )

    print("calculating loss...")
    self.loss = tf.losses.softmax_cross_entropy(self.labels, self.preds, reduction=tf.losses.Reduction.SUM)


    #self.loss = self.loss * -1
    return self.loss

  # BPTT
  def bptt(self):
    """
    section 3.0: https://arxiv.org/pdf/1610.02583.pdf
    """
    print("running bptt...")
    alpha = 10**-4
	
    #Calculate:
    #1) \frac{\partial{Loss}}{dW_{ho}}
    #   = \sum_t \frac{\partial{Loss}}{dz_t} \frac{\partial{z_t}}{dW_{ho}}
    #2) \frac{\partial{Loss}}{db_o}
    #   = \sum_t \frac{\partial{Loss}}{dz_t} \frac{\partial{z_t}}{db_o}
    dLdW_ho = 0.0
    dLdB_o = 0.0
    for i in range(self.t):
      dLdZ_t = tf.gradients(self.loss, self.layers_z[i])[0] # https://stackoverflow.com/a/38033731/1174594
      dZdW_ho = tf.gradients(self.layers_z[i], self.W_ho)[0]
      dZdB_o = tf.gradients(self.layers_z[i], self.b_o)[0]
      print("dLdZ_t: {}".format(dLdZ_t))
      print("dZdW_ho: {}".format(dZdW_ho))
      dLdW_ho += ( dLdZ_t * dZdW_ho )
      dLdB_o  += ( dLdZ_t * dZdB_o )
    self.W_ho -= alpha * dLdW_ho #tf.gradients(self.nll, self.W_ho)[0] # https://stackoverflow.com/a/38033731/1174594
    self.b_o  -= alpha * dLdB_o #tf.gradients(self.nll, self.b_o)[0]

	
    #Calculate:
    #1) \frac{\partial{Loss}}{dW_{hh}}
    #   = \sum_t \sum_k^t \frac{\partial{Loss}}{dz_t} \frac{\partial{z_t}}{dh_t} 
    #                     \frac{\partial{h_t}}{dh_k} \frac{\partial{h_k}}{dW_{hh}}
    #2) \frac{\partial{Loss}}{dW_{ih}}
    #   = \sum_t \sum_k^t \frac{\partial{Loss}}{dz_t} \frac{\partial{z_t}}{dh_t}
    #                     \frac{\partial{h_t}}{dh_k} \frac{\partial{h_k}}{dW_{ih}}
    dLdW_hh = 0.0
    dLdW_ih = 0.0
    for j in range(1, self.t, 1):
      dL_jdW_hh = 0.0
      dL_jdW_ih = 0.0
	
      dL_jdZ_j = tf.gradients(self.loss, self.layers_z[j])[0]
      dZ_jdH_j = tf.gradients(self.loss, self.layers_h[j])[0]
      for k in range(j-1, 0, -1):
        dH_jdH_k = tf.gradients(self.layers_h[j], self.layers_h[k])[0]
        dH_kdW_hh= tf.gradients(self.layers_h[k], self.W_hh)[0]
        dL_jdW_hh += ( dL_jdZ_j * dZ_jdH_j * dH_jdH_k * dH_kdW_hh )
        dL_jdW_ih += ( dL_jdZ_j * dZ_jdH_j * dH_jdH_k * dH_kdW_ih )
      dLdW_hh += dL_jdW_hh
      dLdW_ih += dL_jdW_ih
    self.W_hh -= alpha * dLdW_hh
    self.W_ih -= alpha * dLdW_ih
    #self.b_h  -= # should I neglect this?

  def run_once(self):
    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
      sess.run(init_op)
      with tf.name_scope('Training'):
        sess.run(self.fwd_prop())
        print("fwd done")
        saved_at = saver.save(sess, "arsany/fwd.ckpt")
        print("saved checkpoint at {}".format(saved_at))
        #saver.restore(sess, "arsany/fwd.ckpt")
        #print("successfully restored")
        G_writer = tf.summary.FileWriter('arsany/graph', sess.graph)
        sess.run(self.bptt())
        saved_at = saver.save(sess, "arsany/model.ckpt")
        print("bptt done, model saved at{}".format(saved_at))
        print("W_hh={},W_ih={},b_h={},W_ho={},b_o={}".format(self.W_hh, self.W_ih, self.b_h, self.W_ho, self.b_o))

## load data
data  = np.load("data.npy")[1:]  # ignore first element as it is empty 
labels= np.load("labels.npy")

## cross validation prep
data_train, data_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

model_train = RNNModel(data_train, y_train)
model_train.run_once()
