# Author: Pinyuan Feng
# Created on Dec. 2nd, 2021

import os
import numpy as np
import glob
import csv
import time
import util

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score

DATA_PATH = '../data/'
TEST_RATIO = 0.2
TIMES = 20
MODELS = {
    0: 'LR',
    1: 'KNN',
    2: 'DT',
    3: 'SVM',
    4: 'NBayes',
    5: 'Perceptron',
    6: 'MLP'
}

# Get data
X, y = util.readData(DATA_PATH, "acc+gyro") # "acc"

for id in range(0,len(MODELS)):
    # print(MODELS.get(id,None))

    # Train + Test time
    start = time.perf_counter()

    precision, recall, fscore, accuracy = 0, 0, 0, 0
    test_interval = 0
    for i in range(0,TIMES):
        # Split data
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=TEST_RATIO,shuffle=True) 

        # Data Normalization
        sc = StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)

        # Train
        model = util.classificationModel(MODELS.get(id,None))
        model.fit(X_train_std,y_train)
        
        # Evaluate
        test_s = time.perf_counter()

        y_predict = model.predict(X_test_std)
        p, r, f, _ = score(y_test, y_predict, average='weighted', warn_for=('precision', 'recall', 'f-score'), sample_weight=None, zero_division=0)
        a = accuracy_score(y_test, y_predict)

        test_e = time.perf_counter()

        test_interval += (test_e - test_s)
        accuracy += a
        precision += p
        recall += r 
        fscore += f

    accuracy = accuracy/TIMES
    precision = precision/TIMES
    recall = recall/TIMES
    fscore = fscore/TIMES
    # test_interval = test_interval/TIMES

    end = time.perf_counter()

    interval = end - start

    print('------------------')
    print('Model    : {}'.format(MODELS.get(id,None)))
    print('Accuracy : {:.2%}'.format(accuracy))
    print('F1-score : {:.2%}'.format(fscore))
    print('Precision: {:.2%}'.format(precision))
    print('Recall   : {:.2%}'.format(recall))
    print('CPU Time for training and testing : {:.3} seconds'.format(interval)) # CPU Time for training and testing
    print('CPU Time for testing              : {:.3} seconds'.format(test_interval)) # CPU Time for testing
    print('------------------')
