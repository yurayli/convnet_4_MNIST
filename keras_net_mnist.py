
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
#from keras.optimizers import SGD
import keras.callbacks as kcb


## Read data from CSV file 
train = pd.read_csv("./convnet_MNIST/train.csv").values
train, train_label = train[:,1:]/255., train[:,0]
train = train.astype('float32')
train_label = train_label.astype('int8')
split_size = int(train.shape[0]*0.85)
train_x, val_x = train[:split_size, :], train[split_size:, :]
train_y, val_y = train_label[:split_size].reshape(-1,1), train_label[split_size:].reshape(-1,1)
train_y, val_y = (np.arange(10)==train_y).astype(np.int8), (np.arange(10)==val_y).astype(np.int8)
del train, train_label

# data augmentation
tr_expand_x = np.zeros((4*train_x.shape[0], train_x.shape[1]), dtype='float32')
tr_expand_y = np.zeros((4*train_y.shape[0], train_y.shape[1]), dtype='int8')
i, j = 0, 0
for x, y in zip(train_x, train_y):
	image = np.reshape(x, (-1, 28))
	j += 1
	if j % 1000 == 0: print "Expanding image number", j
	# iterate over data telling us the details of how to
	# do the displacement
	for d, axis in [(2, 0), (-2, 0), (2, 1), (-2, 1)]:
		new_im = np.roll(image, d, axis)
		tr_expand_x[i], tr_expand_y[i] = new_im.reshape(784), y
		i += 1

train_x = np.vstack([train_x, tr_expand_x])
train_y = np.vstack([train_y, tr_expand_y])

# define vars
hidden_num_units = 128
label_units = 10

# create model
model = Sequential([
	Convolution2D(32, 3, 3, init='glorot_uniform', activation='relu', border_mode='same', input_shape=(1,28,28)),
	MaxPooling2D(pool_size=(2,2)),
	Convolution2D(64, 3, 3, init='glorot_uniform', activation='relu'),
	MaxPooling2D(pool_size=(2,2)),
	Dropout(0.3),
	Flatten(),
	Dense(hidden_num_units, init='he_normal', activation='relu'),
	Dropout(0.4),
	Dense(hidden_num_units, init='he_normal', activation='relu'),
	Dropout(0.4),
	Dense(label_units, activation='softmax')
	])

'''
old model (without augmentating data)
model = Sequential([
	Convolution2D(32, 3, 3, init='glorot_uniform', activation='relu', border_mode='same', input_shape=(1,28,28)),
	MaxPooling2D(pool_size=(2,2)),
	Convolution2D(64, 3, 3, init='glorot_uniform', activation='relu', border_mode='same'),
	MaxPooling2D(pool_size=(2,2)),
	Convolution2D(96, 3, 3, init='glorot_uniform', activation='relu', border_mode='same'),
	MaxPooling2D(pool_size=(2,2)),
	Flatten(),
	Dropout(0.2),
	Dense(hidden_num_units*2, init='he_normal', activation='relu'),
	Dropout(0.4),
	Dense(hidden_num_units, init='he_normal', activation='relu'),
	Dropout(0.4),
	Dense(label_units, activation='softmax')
	])
'''

# compile the model with necessary attributes
#opt_sgd = SGD(lr=0.05, decay=0.0002)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

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


epochs = 5
batch_size = 48

print "Start training..."
t0 = time()
calls = Call()
checkpointer = kcb.ModelCheckpoint(filepath="./best_model.hdf5", monitor='val_acc', verbose=1, save_best_only=True)
trained_model = model.fit(train_x.reshape(-1, 1, 28, 28), train_y, nb_epoch=epochs, 
	batch_size=batch_size, validation_data=(val_x.reshape(-1, 1, 28, 28), val_y),
	callbacks=[calls, checkpointer])
print "\nElapsed time:", time()-t0, '\n\n'


model.load_weights('./best_model.hdf5')
pred_tr = model.predict_classes(train_x.reshape(-1, 1, 28, 28))
print "training accuracy:", np.mean(pred_tr==np.argmax(train_y, 1))

pred_val = model.predict_classes(val_x.reshape(-1, 1, 28, 28))
print "validation accuracy:", np.mean(pred_val==np.argmax(val_y, 1))

test = pd.read_csv("./convnet_MNIST/test.csv").values
test = test / 255.
pred = model.predict_classes(test.reshape(-1, 1, 28, 28))

# export Kaggle submission file
fh = open('eval.txt','w+')
fh.write('ImageId,Label\n')
for i in xrange(len(pred)):
    fh.write('%d,%d\n' %(i+1, pred[i]))

fh.close()

# 1172.77 sec, tr_acc=.9980, val_acc=.9921, test_acc=.9900  2 conv layers
# 1344.05 sec, tr_acc=.9952, val_acc=.9933, test_acc=.9919  2 conv layers  5 times training data  4 epochs
# 2024.97 sec, tr_acc=.9963, val_acc=.9933, test_acc=.9929  2 conv layers  5 times training data  6 epochs
# 1756.93 sec, tr_acc=.9968, val_acc=.9937, test_acc=.9940  2 conv layers  5 times training data  5 epochs
# 1524.26 sec, tr_acc=.9954, val_acc=.9913  3 conv layers
