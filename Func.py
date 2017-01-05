## Setting functions 

import numpy as np
import matplotlib.pyplot as plt


def randInitializeWeights(L_in, L_out):
	#epsilonInit = 25/ np.sqrt(L_in + L_out)
	epsilonInit = .12
	W = - epsilonInit + 2 * epsilonInit * np.random.rand(L_out, 1 + L_in)
	return W


def sigmoid(x):
	return 1 / (1 + np.exp(-x))


def forward(Theta, nn_layers_sizes, X, y):
	# Setup some useful variables
	num_layers = len(nn_layers_sizes)
	m = X.shape[0]
	
	# Feed forward
	X = np.hstack([np.ones((m, 1)), X])
	units = []
	units.append( sigmoid(X.dot(Theta[0].T)) )
	units[0] = np.hstack([np.ones((m, 1)), units[0]])
	for l in range(1, num_layers-1):
		units.append( sigmoid(units[l-1].dot(Theta[l].T)) )
		# final output layer need not patch ones
		if l != (num_layers-2):
			units[l] = np.hstack([np.ones((m, 1)), units[l]])
	H = units[-1]  # output hypothesis
	
	# One-hot encoding of labels
	encode_y = np.zeros((m, nn_layers_sizes[-1]))
	for i in range(m):
		encode_y[i,y[i]] += 1
	
	return H, encode_y, units


# Objective function
def costFunc(nn_params, nn_layers_sizes, X, y, lmbda):
	# Setup some useful variables
	m = X.shape[0]
	num_layers = len(nn_layers_sizes)
	
	# Unpack weights parameters nn_params to Theta
	Theta = []
	num_weights = 0
	for i in range(num_layers-1):
		Theta.append(
			nn_params[num_weights:num_weights + nn_layers_sizes[i+1]*(1 + nn_layers_sizes[i])].reshape(
				nn_layers_sizes[i+1], 1 + nn_layers_sizes[i]) )
		num_weights += np.prod(Theta[i].shape)
	
	# Feedforward and return the cost function
	H, encode_y, _ = forward(Theta, nn_layers_sizes, X, y)
	cost = (-encode_y) * np.log(H) - (1-encode_y) * np.log(1-H)
	sumThetaSquare = 0
	for i in range(num_layers-1):
		sumThetaSquare += np.sum(Theta[i][:,1:] ** 2)
	regularFactor = sumThetaSquare * lmbda/(2*m)
	J = (1./m) * np.sum(cost) + regularFactor
	
	return J
	

# Gradient of objective function
def gradFunc(nn_params, nn_layers_sizes, X, y, lmbda):
	# Setup some useful variables
	m = X.shape[0]
	num_layers = len(nn_layers_sizes)
	
	# Unpack weights parameters nn_params to Theta
	Theta = []
	num_weights = 0
	for i in range(num_layers-1):
		Theta.append(
			nn_params[num_weights:num_weights + nn_layers_sizes[i+1]*(1 + nn_layers_sizes[i])].reshape(
				nn_layers_sizes[i+1], 1 + nn_layers_sizes[i]) )
		num_weights += (Theta[i].shape[0]*Theta[i].shape[1])
	
	# To be returned gradients
	Theta_grad = []
	for i in range(num_layers-1):
		Theta_grad.append( np.zeros(Theta[i].shape) )
	
	# Backpropagation algorithm and return the gradient of the cost function
	H, encode_y, units = forward(Theta, nn_layers_sizes, X, y)
	X = np.hstack([np.ones((m, 1)), X])
	delUnits = []
	delUnits.append( H - encode_y )
	Theta_grad[-1] = delUnits[0].T.dot(units[-2])
	for l in range(num_layers-2,0,-1):
		# backward pass error signal, first term of chain rule: (dJ/dz)(dz/dTheta)
		delUnits.insert(0, (delUnits[0].dot(Theta[l]) * (units[l-1]*(1-units[l-1])))[:,1:])

		# second term of chain rule: (dJ/dz)(dz/dTheta)
		if l != 1:
			Theta_grad[l-1] = delUnits[0].T.dot(units[l-2])  # new delhid (i.e., index 0) is required here in each loop
		else:
			Theta_grad[l-1] = delUnits[0].T.dot(X)

	# Regularization
	for i in range(num_layers-1):
		Theta_grad[i] *= (1./m)
		Theta_grad[i][:,1:] += ((lmbda/m) * Theta[i][:,1:])
	
	# Pack gradient of weights Theta_grad to grad
	grad = np.array([])
	for i, T_grad in enumerate(Theta_grad):
		grad = np.hstack([ grad, T_grad.ravel() ])
	
	return grad
	

def predict(Theta, X):
	m = X.shape[0]
	h = sigmoid( np.dot(np.hstack([np.ones((m, 1)), X]), Theta[0].T) )
	for i in range(1, len(Theta)):
		h = sigmoid(np.dot(np.hstack([np.ones((m,1)), h]), Theta[i].T))
	
	#prob = np.max(h, axis=1)
	pred = h.argmax(1)
	'''
	pred = np.array([])
	for i in range(m):
		pred = np.append( pred, np.where(h[i] == np.max(h[i]))[0] )
	'''
	return pred
	


def displayData(X):
	
	# Compute rows, cols of image
	m, n = X.shape[0], X.shape[1]
	sample_width = ( np.round(np.sqrt(n)) ).astype('int')
	sample_height = ( n /sample_width ).astype('int')
	
	# Compute number of items to display
	display_rows = np.floor(np.sqrt(m))
	display_cols = np.ceil(m /display_rows)
	
	# Padding between images
	pad = 1.
	
	# Setup blank display
	display_array = - np.ones( (pad + display_rows * (sample_height + pad), 
								pad + display_cols * (sample_width + pad)) )
	
	# copy each example into a patch on the display array
	curr_ex = 0
	for i in range(display_rows.astype('int')):
		for j in range(display_cols.astype('int')):
			if curr_ex > m:
				break
			
			max_val = np.max(abs(X[curr_ex, :]))
			display_array[
			pad + i * (sample_height + pad):pad + i * (sample_height + pad) + sample_height, 
			pad + j * (sample_width + pad):pad + j * (sample_width + pad) + sample_width] = \
			X[curr_ex, :].reshape(sample_height, sample_width) / max_val
			
			curr_ex += 1
		if curr_ex > m:
			break
	
	display_array = display_array.astype('float')
	plt.imshow(display_array, cmap='gray', aspect='equal')
	plt.axis('off')
	plt.show()


	
	
	

	
