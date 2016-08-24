## Using Convolutional and Fully-connected Neural Nets 
## to train MNIST digits dataset
## Using mini-batch SGD in optimization

## Libraries
# Standard library
import six.moves.cPickle as pickle

# Third-party libraries
import pandas as pd
import numpy as np
import theano
import theano.tensor as T
import convnet as cn
from time import time


## Read data from CSV file 
train = pd.read_csv("./train.csv").values

## Setting features and labels
Xval, yval = train[:4000,1:], train[:4000,0]
X, y = train[4000:,1:], train[4000:,0]
Xval = Xval / 255.
X = X / 255.

def shared(data):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.

        """
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")

train_data, valid_data = shared((X, y)), shared((Xval, yval))


### Experiment 1: sigmoid
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
net.SGD(train_data, num_epochs, mini_batch_size, eta, valid_data, early_stop=True)
time() - t0
# 2302.698 sec, 99.79% train accu, 98.87% val accu, 98.94% test accu

### Experiment 2: ReLU
num_epochs = 60
mini_batch_size = 10
eta = 0.05
net = cn.Network([
        cn.ConvPoolLayer(image_shape=(1, 28, 28), 
                      filter_shape=(20, 1, 5, 5), 
                      poolsize=(2, 2),
                      activation_fn=cn.ReLU),
        cn.ConvPoolLayer(image_shape=(20, 12, 12), 
                      filter_shape=(40, 20, 5, 5), 
                      poolsize=(2, 2),
                      activation_fn=cn.ReLU),
        cn.FullyConnectedLayer(n_in=40*4*4, n_out=100,
                      activation_fn=cn.ReLU),
        cn.SoftmaxLayer(n_in=100, n_out=10)])
t0 = time()
net.SGD(train_data, num_epochs, mini_batch_size, eta, valid_data, lmbda=0.1)
time() - t0
# 6029.097 sec, 100.00% train accu, 99.22% val accu, 99.01% test accu

# Calculate training accuracy
train_x, train_y = train_data
num_train_batches = cn.size(train_data)/mini_batch_size
i = T.lscalar()
train_mb_accuracy = theano.function(
    [i], net.layers[-1].accuracy(net.y),
    givens={
        net.x:
        train_x[i*mini_batch_size: (i+1)*mini_batch_size],
        net.y:
        train_y[i*mini_batch_size: (i+1)*mini_batch_size]
        })

train_accuracy = np.mean(
    [train_mb_accuracy(j) for j in xrange(num_train_batches)])



## Write the evaluation of testset into file
test = pd.read_csv("./test.csv").values
test = test / 255.
test_data = theano.shared(np.asarray(test, 
    dtype=theano.config.floatX), borrow=True)
net = pickle.load(open('best_model.pkl'))
t1 = time()
pred = net.predict(test_data)
# export Kaggle submission file
fh = open('eval.txt','w+')
fh.write('ImageId,Label\n')
for i in xrange(len(pred)):
    fh.write('%d,%d\n' %(i+1, pred[i]))

fh.close()
print '\nElapsed time:', time()-t1, 'seconds.\n'
