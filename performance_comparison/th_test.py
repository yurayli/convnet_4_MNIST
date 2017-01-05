## Using Convolutional and Fully-connected Neural Nets 
## to train MNIST digits dataset

## Libraries
# Standard library
import six.moves.cPickle as pickle

# Third-party libraries
import pandas as pd
import numpy as np
import theano
import theano.tensor as T
import cnet as cn
from time import time


## Read data from CSV file 
train = pd.read_csv("./convnet_MNIST/train.csv").values

## Setting features and labels
Xval, yval = train[20000:25000,1:], train[20000:25000,0]
X, y = train[:20000,1:], train[:20000,0]
Xval = Xval / 255.
X = X / 255.
del train

def shared(data):
    """Place the data into shared variables.  This allows Theano to copy
       the data to the GPU, if one is available.

    """
    shared_x = theano.shared(
      np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
    shared_y = theano.shared(
      np.asarray(data[1], dtype='int32'), borrow=True)
    return shared_x, shared_y

train_data, valid_data = shared((X, y)), shared((Xval, yval))
del X, Xval, y, yval


### Experiment 5:
num_epochs = 15
mini_batch_size = 32
eta = 0.05
net = cn.Network([
    cn.ConvPoolLayer(image_shape=(1, 28, 28), 
                  filter_shape=(16, 1, 5, 5), 
                  poolsize=(2, 2),
                  activation_fn=cn.ReLU),
    cn.ConvPoolLayer(image_shape=(16, 12, 12), 
                  filter_shape=(32, 16, 5, 5), 
                  poolsize=(2, 2),
                  activation_fn=cn.ReLU),
    cn.FullyConnectedLayer(n_in=32*4*4, n_out=128,
                  activation_fn=cn.ReLU, p_dropout=0.5),
    cn.FullyConnectedLayer(n_in=128, n_out=128,
                  activation_fn=cn.ReLU, p_dropout=0.5),
    cn.SoftmaxLayer(n_in=128, n_out=10)])
t0 = time()
net.fit(train_data, num_epochs, mini_batch_size, eta, valid_data, lmbda=0.005, optim_mode='adam')
print "\nElapsed time:", time() - t0
# 1393.87 sec, 99.495% train accu, 99.07% val accu, 99.057% test accu
# 1451.25 sec, 99.08% train accu, 98.99% val accu, 98.84% test accu, 20 epochs
# 769.010 sec, 99.14% train accu, 98.83% val accu, ?????? test accu, 10 epochs

pred_tr = net.predict(train_data[0])
print "Training accuracy:", np.mean(pred_tr==train_data[1].get_value())

pred_val = net.predict(valid_data[0])
print "Validation accuracy:", np.mean(pred_val==valid_data[1].get_value())


