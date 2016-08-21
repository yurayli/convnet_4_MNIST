## Using fully-connected Neural Network to train MNIST digits dataset
## Using scipy.optimize package to optimize the NNet

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import fmin_cg
import matplotlib.pyplot as plt
from time import time
from Func import *


## Setup the parameters
input_layer_size  = 784   # 28x28 Input Images of Digits
hid_layer1_size = 100
hid_layer2_size = 49
hid_layer3_size = 25
num_labels = 10           # 10 labels, from 0 to 9
                          
nn_layers_sizes = [input_layer_size, hid_layer1_size, hid_layer2_size, num_labels]
num_layers = len(nn_layers_sizes)

## read data from CSV file 
train = pd.read_csv("./train.csv").values

## setting features and labels
Xval, yval = train[:4000,1:], train[:4000,0]
X, y = train[4000:,1:], train[4000:,0]
Xval = (Xval-128.) / 128.
X = (X-128.) / 128.

## Visualization
print 'Loading and Visualizing Data ...\n'
m = X.shape[0]
sel = np.random.permutation(m)
sel = sel[0:100]

displayData(X[sel, :])

print 'Program paused. Press enter to continue.\n'
raw_input()


## Initializing parameters
print '\nInitializing Neural Network Parameters ...\n'

initial_Theta = []
for i in range(num_layers-1):
	initial_Theta.append( randInitializeWeights(nn_layers_sizes[i], nn_layers_sizes[i+1]) )

# Unroll parameters
initial_nn_params = np.array([])
for i, Theta in enumerate(initial_Theta):
	initial_nn_params = np.hstack([ initial_nn_params, Theta.ravel() ])


## Training NN
print '\nTraining Neural Network... \n'

lamb = .1
numOfIter = 500

# Create "short hand" for the cost function to be minimized
iter = 1
cost_train = []
cost_val = []
max_accu_val = 0
def call(nn_params):
	global iter
	global nn_layer_sizes
	global cost_train
	global cost_val
	global max_accu_val
	Theta = []
	num_weights = 0
	for i in range(num_layers-1):
		if i == 0:
				Theta.append( np.reshape(nn_params[0:nn_layers_sizes[i+1]*(1 + nn_layers_sizes[i])], 
							  (nn_layers_sizes[i+1], 1 + nn_layers_sizes[i])) )
		else:
			Theta.append( np.reshape(nn_params[num_weights:num_weights + nn_layers_sizes[i+1]*(1 + nn_layers_sizes[i])], 
						  (nn_layers_sizes[i+1], 1 + nn_layers_sizes[i])) )
		num_weights += (Theta[i].shape[0]*Theta[i].shape[1])
	pred_train = predict(Theta, X)
	pred_val = predict(Theta, Xval)
	accu_train = np.mean((pred_train == y.ravel()).astype(float)) * 100
	accu_val = np.mean((pred_val == yval.ravel()).astype(float)) * 100
	
	cost_train.append(costFunc(nn_params, nn_layers_sizes, X, y, lamb))
	cost_val.append(costFunc(nn_params, nn_layers_sizes, Xval, yval, lamb))
	print '\nIter {0}: accu_train = {1}, accu_val = {2}'.format(iter, \
		accu_train, accu_val)
	if accu_val > max_accu_val:
		max_accu_val = accu_val
		print '\nThe best validation accuracy so far is {}'.format(max_accu_val)
	iter += 1

'''
t0 = time()
OptResult = fmin_cg(costFunc, initial_nn_params, maxiter=100, fprime=gradFunc, 
					 args=(nn_layers_sizes, X, y, lamb), full_output=True)
print '\nElapsed time of training:', time()-t0, 'seconds.\n'

nn_params = OptResult[0]
cost = OptResult[1]
'''

t0 = time()	
OptResult = minimize(costFunc, initial_nn_params, method='CG', jac=gradFunc, callback=call, 
					 args=(nn_layers_sizes, X, y, lamb), options={'maxiter': numOfIter})
print '\nElapsed time of training:', time()-t0, 'seconds.\n'

nn_params = OptResult.x
plt.plot(range(numOfIter), cost_train, range(numOfIter), cost_val)
plt.legend(('J_train', 'J_val'), loc='best')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.show()

# Unpack weights parameters nn_params to Theta
Theta = []
num_weights = 0
for i in range(num_layers-1):
	if i == 0:
			Theta.append( np.reshape(nn_params[0:nn_layers_sizes[i+1]*(1 + nn_layers_sizes[i])], 
						  (nn_layers_sizes[i+1], 1 + nn_layers_sizes[i])) )
	else:
		Theta.append( np.reshape(nn_params[num_weights:num_weights + nn_layers_sizes[i+1]*(1 + nn_layers_sizes[i])], 
					  (nn_layers_sizes[i+1], 1 + nn_layers_sizes[i])) )
	num_weights += (Theta[i].shape[0]*Theta[i].shape[1])

print 'Program paused. Press enter to continue.\n'
raw_input()


## Implement predict
t1 = time()
pred = predict(Theta, X)
print '\nTraining Set Accuracy: %f' %(np.mean((pred == y.ravel()).astype(float)) * 100)
print '\nElapsed time of evaluating train set:', time()-t1, 'seconds.\n'

print '\nImport test data and evaluating...(will be written in eval.txt)\n'
t2 = time()
test = pd.read_csv("./test.csv").values
test = (test-128.) / 128.
pred = predict(Theta, test)
# accu = 0.9996 for 4 layers with 500 iter and elapsed training time 1452.52 sec, [784 100 49 10]
#
# accu_val = 0.9638 for 4 layers with 500 iter, lamb = 1.
# and elapsed training time xx sec, [784 100 49 10]
#
# accu_val = 0.968 for 4 layers with 500 iter, lamb = .1
# and elapsed training time xx sec, [784 100 49 10]
#
# accu = 0.9999 for 5 layers with 1000 iter and elapsed training time 2690.88 sec, [784 81 49 25 10]
# accu = 0.9960 for 5 layers with 500 iter and elapsed training time xx sec, [784 81 49 25 10]

# export Kaggle submission file
fh = open('eval.txt','w+')
fh.write('ImageId,Label\n')
for i in range(len(pred)):
    fh.write('%d,%d\n' %(i+1, pred[i]))
fh.close()
print '\nElapsed time:', time()-t2, 'seconds.\n'





