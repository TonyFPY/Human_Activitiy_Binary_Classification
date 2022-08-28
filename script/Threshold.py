# Author: Pinyuan Feng
# Created on Dec. 2nd, 2021

import os
import numpy as np
import glob
import time
import util

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score

DATA_PATH = '../data/'
TEST_RATIO = 0.2
TIMES = 20
ACC_THRESHOLD = 0.3
GYRO_THRESHOLD = 0.005
DATA_TYPE = "accGyro"

# Get data
X, y = util.readData(DATA_PATH, DATA_TYPE)
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=TEST_RAIIO) 

# print(X[0:10,0:7])

# Evaluation
start = time.perf_counter()

precision, recall, fscore = 0, 0, 0
for i in range(0,TIMES):
    if DATA_TYPE == "acc":
        y_predict = np.any(X[:,0:3]>=ACC_THRESHOLD , axis=1).astype(int)
    else:
        conAcc = np.any(X[:,0:3]>=ACC_THRESHOLD , axis=1)
        conGyro = np.any(X[:,3:5]>=GYRO_THRESHOLD, axis=1)
        y_predict = [x or y for x,y in zip(conAcc,conGyro)]
        y_predict = np.array(y_predict).astype(int)

    p, r, f, _ = score(y, y_predict, average='weighted', warn_for=('precision', 'recall', 'f-score'), sample_weight=None, zero_division=0)
    
    precision += p
    recall += r 
    fscore += f

precision = precision/TIMES
recall = recall/TIMES
fscore = fscore/TIMES

end = time.perf_counter()
interval = end - start
accuracy = accuracy_score(y, y_predict)

print('------------------')
print('Accuracy : {:.2%}'.format(accuracy))
print('F1-score : {:.2%}'.format(fscore))
print('Precision: {:.2%}'.format(precision))
print('Recall   : {:.2%}'.format(recall))
print('CPU time : {:.3} seconds'.format(interval))
print('------------------')