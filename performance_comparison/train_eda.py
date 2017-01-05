
accu_th = [94.87, 96.31, 96.96, 97.78, 97.76, 97.76, 98.04, 98.20, 98.08, 98.34, 98.36, 98.54, 98.62, 98.50, 98.62]
accu_tf = [96.28, 97.56, 98.00, 97.64, 98.36, 97.72, 97.82, 97.98, 98.04, 98.18, 98.44, 97.94, 98.46, 98.56, 98.36]
accu_k_th = [96.72, 97.66, 97.62, 98.22, 98.28, 98.48, 98.40, 98.56, 98.22, 98.62, 98.30, 98.66, 98.20, 98.58, 98.56]
accu_k_tf = [97.22, 97.42, 98.00, 97.82, 98.30, 98.12, 98.30, 98.40, 98.62, 98.68, 98.56, 98.40, 98.64, 98.46, 98.60]

import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.figure(figsize=(8,5))
plt.plot(range(1,16), accu_th, '-o', label='theano')
plt.plot(range(1,16), accu_tf, '-o', label='tensorflow')
plt.plot(range(1,16), accu_k_th, '-o', label='keras_th based')
plt.plot(range(1,16), accu_k_tf, '-o', label='keras_tf based')
plt.xlim(0,16)
plt.xlabel('Epoch')
plt.ylabel('Valid accuracy')
plt.legend(loc='lower right')
plt.savefig('./comparison.png')



# ==================================================================
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

with open('opts_comparison.pkl') as f:
	opt_dicts, spend_time = pickle.load(f)

optimizers = ['sgd', 'adagrad', 'adadelta', 'rmsprop', 'adam', 'adamax', 'nadam']

ind = np.arange(len(spend_time))  # the x locations for the groups
width = 0.4       # the width of the bars
plt.figure(figsize=(8,5))
plt.bar(ind+0.5*width, spend_time.values(), width)
plt.xlabel('Optimizers')
plt.ylabel('Training time')
plt.xticks(ind+width, spend_time.keys())
plt.savefig('./runtime_comparison.png')

plt.figure(figsize=(8,5))
#plt.plot(range(1,11), opt_dicts['sgd'], '-o', label='sgd')
plt.plot(range(1,11), opt_dicts['adagrad'], '-o', label='adagrad')
plt.plot(range(1,11), opt_dicts['adadelta'], '-o', label='adadelta')
plt.plot(range(1,11), opt_dicts['rmsprop'], '-o', label='rmsprop')
plt.plot(range(1,11), opt_dicts['adam'], '-o', label='adam')
plt.plot(range(1,11), opt_dicts['adamax'], '-o', label='adamax')
plt.plot(range(1,11), opt_dicts['nadam'], '-o', label='nadam')
plt.xlim(0,11)
plt.xlabel('Epochs')
plt.ylabel('Valid accuracy')
plt.legend(loc='lower right')
plt.savefig('./performance_comparison.png')

