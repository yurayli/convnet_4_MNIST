
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
from keras.optimizers import SGD
import keras.callbacks as kcb


## Read data from CSV file 
train = pd.read_csv("./convnet_MNIST/train.csv").values
train, train_label = train[:,1:]/255., train[:,0]
#shuffle = np.random.permutation(train.shape[0])
#train, train_label = train[[shuffle]], train_label[shuffle]
split_size = int(train.shape[0]*0.85)
train_x, val_x = train[:split_size, :], train[split_size:, :]
train_y, val_y = train_label[:split_size].reshape(-1,1), train_label[split_size:].reshape(-1,1)
train_y, val_y = (np.arange(10)==train_y).astype(np.float32), (np.arange(10)==val_y).astype(np.float32)

# define vars
hidden_num_units = 128
label_units = 10

# create model
#model = Sequential([
#  Dense(output_dim=hidden_num_units, input_dim=input_num_units, activation='relu'),
#  Dense(output_dim=label_units, input_dim=hidden_num_units, activation='softmax'),
#])

model = Sequential([
	Convolution2D(32, 3, 3, activation='relu', border_mode='same', input_shape=(1,28,28)),
	MaxPooling2D(pool_size=(2,2)),
	Convolution2D(64, 3, 3, activation='relu', border_mode='same'),
	MaxPooling2D(pool_size=(2,2)),
	Convolution2D(64, 3, 3, activation='relu', border_mode='same'),
	MaxPooling2D(pool_size=(2,2)),
	Dropout(0.2),
	Flatten(),
	Dense(hidden_num_units*2, activation='relu'),
	Dropout(0.4),
	Dense(hidden_num_units, activation='relu'),
	Dropout(0.4),
	Dense(label_units, activation='softmax')
	])

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
#        	with open("best_model.pkl", "wb") as f:
#        		pickle.dump(self.model, f)

epochs = 12
batch_size = 64

print "Start training..."
t0 = time()
calls = Call()
checkpointer = kcb.ModelCheckpoint(filepath="./best_model.hdf5", verbose=1, save_best_only=True)
trained_model = model.fit(train_x.reshape(-1, 1, 28, 28), train_y, nb_epoch=epochs, 
	batch_size=batch_size, validation_data=(val_x.reshape(-1, 1, 28, 28), val_y),
	callbacks=[calls, checkpointer])
print "\nElapsed time:", time()-t0, '\n\n'


model.load_weights('./best_model.hdf5')
pred_tr = model.predict_classes(train_x.reshape(-1, 1, 28, 28))
print "training accuracy:", sum(pred_tr==np.argmax(train_y, 1))/float(train_x.shape[0])

pred_val = model.predict_classes(val_x.reshape(-1, 1, 28, 28))
print "validation accuracy:", sum(pred_val==np.argmax(val_y, 1))/float(val_x.shape[0])

test = pd.read_csv("./convnet_MNIST/test.csv").values
test = test / 255.
pred = model.predict_classes(test.reshape(-1, 1, 28, 28))

# export Kaggle submission file
fh = open('eval.txt','w+')
fh.write('ImageId,Label\n')
for i in xrange(len(pred)):
    fh.write('%d,%d\n' %(i+1, pred[i]))

fh.close()

# 1172.77 sec, tr_acc=.9980, val_acc=.9921, test_acc=.9900
