## ConvNet in theano
## with the MNIST digits dataset

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


# Global Contrast Normalization
def norm_input(x): return (x-mean_px)/std_px

# Transform data to theano's format
def shared(data):
    """Place the data into shared variables.  This allows Theano to copy
       the data to the GPU, if one is available.
    """
    shared_x = theano.shared(
      np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
    shared_y = theano.shared(
      np.asarray(data[1], dtype='int32'), borrow=True)
    return shared_x, shared_y


## Load data from CSV file 
train = pd.read_csv("./convnet_theano/train.csv").values

## Setting features and labels
train, train_label = train[:,1:], train[:,0]
mean_px, std_px = train.mean(), train.std()
train = norm_input(train)
Xval, yval = train[:6320], train_label[:6320]
X, y = train[6320:], train[6320:]
del train, train_label

train_data, valid_data = shared((X, y)), shared((Xval, yval))
del X, Xval


## Training
# Experiment 1: sigmoid
num_epochs = 60
mini_batch_size = 10
eta = 0.1
net = cn.Network([
    cn.ConvPoolLayer(image_shape=(1, 28, 28), 
                  filter_shape=(20, 1, 5, 5), 
                  poolsize=(2, 2)),
    cn.ConvPoolLayer(image_shape=(20, 12, 12), 
                  filter_shape=(40, 20, 5, 5), 
                  poolsize=(2, 2)),
    cn.FullyConnectedLayer(n_in=40*4*4, n_out=100),
    cn.SoftmaxLayer(n_in=100, n_out=10)])
t0 = time()
net.fit(train_data, num_epochs, mini_batch_size, eta, valid_data, early_stop=True)
time() - t0
# 99.79% train accu, 98.87% val accu, 98.94% test accu

# Experiment 2: relu
num_epochs = 20
mini_batch_size = 16
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
net.fit(train_data, num_epochs, mini_batch_size, eta, valid_data, 
  lmbda=0.005, optim_mode='adam')
print "Training elapsed time:", time() - t0
# 99.495% train accu, 99.07% val accu, 99.057% test accu


## Validating
# Calculate training accuracy
print "Calculating training accuracy..."
train_x, train_y = train_data
pred_trn = net.predict(train_x)
train_acc = np.mean(pred_trn == y)
print "Training accuracy:", train_acc

# Calculate validating accuracy
print "Calculating validating accuracy..."
valid_x, valid_y = valid_data
pred_val = net.predict(valid_x)
valid_acc = np.mean(pred_val == yval)
print "Training accuracy:", valid_acc


## Write the evaluation of testset into file
test = pd.read_csv("./convnet_theano/test.csv").values
test = norm_input(test)
test_data = theano.shared(np.asarray(test, 
    dtype=theano.config.floatX), borrow=True)
net = pickle.load(open('./best_model.pkl'))
print "\nEvaluating..."
t1 = time()
pred = net.predict(test_data)

# export Kaggle submission file
fh = open('eval.txt','w+')
fh.write('ImageId,Label\n')
for i in xrange(len(pred)):
    fh.write('%d,%d\n' %(i+1, pred[i]))

fh.close()
print '\nElapsed time:', time()-t1, 'seconds.\n'
