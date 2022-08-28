# Author: Pinyuan Feng
# Created on Dec. 3rd, 2021

import os
import numpy as np
import glob
import csv
import time
import util

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score

DATA_PATH = '../data/'
GRAPH_PATH = '../graph/'
TEST_RATIO = 0.2
TIMESTAMPS = 20
DATA_TYPE = "acc+gyro"
NUM_OF_FEATURES = 3 if DATA_TYPE == "acc" else 6
TIMES = 10

# Get data
X, y = util.readTimeSeriesData(DATA_PATH,TIMESTAMPS,DATA_TYPE)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=TEST_RATIO,shuffle=True) 

# Data normalization for multi-dimensional data
X_train = X_train.reshape(-1,NUM_OF_FEATURES)
X_test = X_test.reshape(-1,NUM_OF_FEATURES)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

X_train_std = X_train_std.reshape(-1,TIMESTAMPS,NUM_OF_FEATURES)
X_test_std = X_test_std.reshape(-1,TIMESTAMPS,NUM_OF_FEATURES)

# Create the model and train it
model = util.buildModel((TIMESTAMPS, NUM_OF_FEATURES), "LSTM") # "BD_LSTM"
history = model.fit(X_train_std, y_train, epochs=20, validation_split=0.25, verbose=1)

acc_values = history.history['acc']
val_acc_values = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(acc_values)+1)

# show training loss and validating loss
plt.subplot(1, 2, 1)
plt.plot(epochs,loss,'r',label='trainning loss')
plt.plot(epochs,val_loss,'b',label='validating loss',linestyle='--')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# show training and validating accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs,acc_values,'r',label='Training accuracy')
plt.plot(epochs,val_acc_values,'b',label='validating accuracy',linestyle='--')
plt.title('Training and validating accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig(GRAPH_PATH + "LSTM.png")
plt.show()

# Test
start = time.perf_counter()

y_predict = model.predict(X_test_std, verbose=0)
y_predict = np.squeeze(np.where(y_predict >= 0.5, 1, 0))
p, r, f, _ = score(y_test, y_predict, average='weighted', warn_for=('precision', 'recall', 'f-score'), sample_weight=None, zero_division=0)

end = time.perf_counter()

a = accuracy_score(y_test, y_predict)

print('------------------')
print('Accuracy : {:.2%}'.format(a))
print('F1-score : {:.2%}'.format(f))
print('Precision: {:.2%}'.format(p))
print('Recall   : {:.2%}'.format(r))
print('CPU time : {:.2} seconds'.format(end-start))
print('------------------')