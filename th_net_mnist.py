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
Xval, yval = train[:6320,1:], train[:6320,0]
X, y = train[6320:,1:], train[6320:,0]
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
net.fit(train_data, num_epochs, mini_batch_size, eta, valid_data, lmbda=0.005, optim_mode='adam')
print "Training elapsed time:", time() - t0
# 1393.87 sec, 99.495% train accu, 99.07% val accu, 99.057% test accu
# 1451.25 sec, 99.08% train accu, 98.99% val accu, 98.84% test accu, 20 epochs
# 769.010 sec, 99.14% train accu, 98.83% val accu, ?????? test accu, 10 epochs


'''
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
net.fit(train_data, num_epochs, mini_batch_size, eta, valid_data, early_stop=True)
time() - t0
# 2302.698 sec, 99.79% train accu, 98.87% val accu, 98.94% test accu

### Experiment 2: ReLU and Dropout
num_epochs = 40
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
                  activation_fn=cn.ReLU, p_dropout=0.5),
    cn.SoftmaxLayer(n_in=100, n_out=10, p_dropout=0.5)])
t0 = time()
net.fit(train_data, num_epochs, mini_batch_size, eta, valid_data, lmbda=0.1)
time() - t0
# 3972.386 sec, 99.73% train accu, 99.32% val accu, 99.014% test accu

### Experiment 3:
num_epochs = 20
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
                  activation_fn=cn.ReLU, p_dropout=0.5),
    cn.SoftmaxLayer(n_in=100, n_out=10, p_dropout=0.5)])
t0 = time()
net.fit(train_data, num_epochs, mini_batch_size, eta, valid_data, lmbda=0.001)
time() - t0
# 1826.49 sec, 99.52% train accu, 99.22% val accu, 98.986% test accu

### Experiment 4:
num_epochs = 25
mini_batch_size = 8
eta = 0.05
net = cn.Network([
    cn.ConvPoolLayer(image_shape=(1, 28, 28), 
                  filter_shape=(16, 1, 5, 5), 
                  poolsize=(2, 2),
                  activation_fn=cn.ReLU),
    cn.ConvPoolLayer(image_shape=(16, 12, 12), 
                  filter_shape=(40, 16, 5, 5), 
                  poolsize=(2, 2),
                  activation_fn=cn.ReLU),
    cn.FullyConnectedLayer(n_in=40*4*4, n_out=128,
                  activation_fn=cn.ReLU, p_dropout=0.5),
    cn.FullyConnectedLayer(n_in=128, n_out=128,
                  activation_fn=cn.ReLU, p_dropout=0.5),
    cn.SoftmaxLayer(n_in=128, n_out=10)])
t0 = time()
net.fit(train_data, num_epochs, mini_batch_size, eta, valid_data, lmbda=0.005)
print "Training elapsed time:", time() - t0
# 2036.54 sec, 99.246% train accu, 99.03% val accu, 99.057% test accu
'''

# Calculate training accuracy
print "Calculating training accuracy..."
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
print "Training accuracy:", train_accuracy



## Write the evaluation of testset into file
test = pd.read_csv("./convnet_MNIST/test.csv").values
test = test / 255.
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
