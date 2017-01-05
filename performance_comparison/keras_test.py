
## Libraries
#import os
import numpy as np
import pandas as pd
import h5py
from time import time
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.regularizers import l2
import keras.callbacks as kcb


## Read data from CSV file 
train = pd.read_csv("./convnet_MNIST/train.csv").values
train, train_label = train[:,1:]/255., train[:,0]
train = train.astype('float32')
train_label = train_label.astype('int8')
split_size = 20000  #int(train.shape[0]*0.85)
train_x, val_x = train[:split_size, :], train[split_size:split_size+5000, :]
train_y, val_y = train_label[:split_size].reshape(-1,1), train_label[split_size:split_size+5000].reshape(-1,1)
train_y, val_y = (np.arange(10)==train_y).astype(np.int8), (np.arange(10)==val_y).astype(np.int8)
del train, train_label


# define vars
hidden_num_units = 128
label_units = 10

# create model
model = Sequential([
	Convolution2D(16, 5, 5, init='normal', activation='relu', border_mode='valid', input_shape=(1,28,28)),
	MaxPooling2D(pool_size=(2,2)),
	Convolution2D(32, 5, 5, init='normal', activation='relu', border_mode='valid'),
	MaxPooling2D(pool_size=(2,2)),
	Flatten(),
	Dense(hidden_num_units, init='he_normal', activation='relu'),
	Dense(hidden_num_units, init='he_normal', activation='relu'),
	Dense(label_units, activation='softmax')
	])



# compile the model with necessary attributes
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

## Training!!
epochs = 15
batch_size = 32

print "Start training..."
t0 = time()
trained_model = model.fit(train_x.reshape(-1, 1, 28, 28), train_y, nb_epoch=epochs, 
	batch_size=batch_size, validation_data=(val_x.reshape(-1, 1, 28, 28), val_y))
print "\nElapsed time:", time()-t0, '\n\n'

pred_tr = model.predict_classes(train_x.reshape(-1, 1, 28, 28))
print "training accuracy:", np.mean(pred_tr==np.argmax(train_y, 1))

pred_val = model.predict_classes(val_x.reshape(-1, 1, 28, 28))
print "validation accuracy:", np.mean(pred_val==np.argmax(val_y, 1))


