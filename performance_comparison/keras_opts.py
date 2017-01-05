
## Libraries
#import os
import numpy as np
import pandas as pd
import h5py
import cPickle as pickle
from time import time
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
import keras.callbacks as kcb


## Read data from CSV file 
train = pd.read_csv("./convnet_MNIST/train.csv").values
train, train_label = train[:,1:]/255., train[:,0]
#shuffle = np.random.permutation(train.shape[0])
#train, train_label = train[[shuffle]], train_label[shuffle]
split_size = int(train.shape[0]*0.25)
train_x, val_x = train[:split_size, :], train[split_size:split_size*2, :]
train_y, val_y = train_label[:split_size].reshape(-1,1), train_label[split_size:split_size*2].reshape(-1,1)
train_y, val_y = (np.arange(10)==train_y).astype(np.float32), (np.arange(10)==val_y).astype(np.float32)

# define vars
hidden_num_units = 128
label_units = 10



## Training!!
class Call(kcb.Callback):
    def on_train_begin(self, logs={}):
        self.best_acc = 0.0
        self.accs = []
        self.val_accs = []
        self.losses = []
        self.val_losses = []
    def on_epoch_end(self, batch, logs={}):
    	self.accs.append(logs.get('acc'))
    	self.val_accs.append(logs.get('val_acc'))
    	self.losses.append(logs.get('loss'))
    	self.val_losses.append(logs.get('val_loss'))
    	if logs.get('val_acc') > self.best_acc:
    		self.best_acc = logs.get('val_acc')
    		print "\nThe BEST val_acc to date."

epochs = 10
batch_size = 48
optimizers = ['sgd', 'adagrad', 'adadelta', 'rmsprop', 'adam', 'adamax', 'nadam']
opt_dicts = dict()
spend_time = dict()

print "Start training..."
t0 = time()

for opt in optimizers:
	tStart = time()

	# create model
	model = Sequential([
		Convolution2D(16, 3, 3, init='glorot_normal', activation='relu', border_mode='same', input_shape=(1,28,28)),
		MaxPooling2D(pool_size=(2,2)),
		Convolution2D(32, 3, 3, init='glorot_normal', activation='relu', border_mode='same'),
		MaxPooling2D(pool_size=(2,2)),
		Flatten(),
		Dropout(0.2),
		Dense(hidden_num_units, init='he_normal', activation='relu'),
		Dropout(0.4),
		Dense(hidden_num_units, init='he_normal', activation='relu'),
		Dropout(0.4),
		Dense(label_units, activation='softmax')
	])

	# compile the model with necessary attributes
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

	calls = Call()
	trained_model = model.fit(train_x.reshape(-1, 1, 28, 28), train_y, nb_epoch=epochs, 
		batch_size=batch_size, validation_data=(val_x.reshape(-1, 1, 28, 28), val_y),
		callbacks=[calls])
	opt_dicts[opt] = calls.val_accs
	spend_time[opt] = time()-tStart
	print "\nElapsed time:", spend_time[opt], '\n\n'

print "\nTotal elapsed time:", time()-t0, '\n\n'
with open("opts_comparison.pkl", "wb") as f:
	pickle.dump([opt_dicts, spend_time], f)
'''
model.load_weights('./best_model.hdf5')
pred_tr = model.predict_classes(train_x.reshape(-1, 1, 28, 28))
print "training accuracy:", sum(pred_tr==np.argmax(train_y, 1))/float(train_x.shape[0])

pred_val = model.predict_classes(val_x.reshape(-1, 1, 28, 28))
print "validation accuracy:", sum(pred_val==np.argmax(val_y, 1))/float(val_x.shape[0])
'''

# 1172.77 sec, tr_acc=.9980, val_acc=.9921, test_acc=.9900
# 1524.26 sec, tr_acc=.9954, val_acc=.9913
