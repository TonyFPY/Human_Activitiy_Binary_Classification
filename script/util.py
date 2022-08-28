# Author: Pinyuan Feng
# Created on Dec. 2nd, 2021

import os
import numpy as np
import glob
import csv

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier

from keras.layers import Bidirectional
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers import LeakyReLU

def readData(DATA_PATH, dataType):

    LABELS = ['0','1']
    if dataType == "acc":
        
        X = np.empty((0,3))
        y = np.empty((0))
        
        for label in LABELS:
            csvPath = os.path.join(DATA_PATH,label)
            filenames = glob.glob(csvPath + "/*.csv")
            for filename in filenames:
                # print(filename)
                with open(filename,'r') as csvfile:
                    content = csv.reader(csvfile)
                    for row in content:
                        X = np.append(X, np.array([[float(row[0]),float(row[1]),float(row[2])]]), axis=0)
                        y = np.append(y, np.array([int(label)]), axis=0)
    else:

        X = np.empty((0,6))
        y = np.empty((0))
        
        for label in LABELS:
            csvPath = os.path.join(DATA_PATH,label)
            filenames = glob.glob(csvPath + "/*.csv")
            for filename in filenames:
                # print(filename)
                with open(filename,'r') as csvfile:
                    content = csv.reader(csvfile)
                    for row in content:
                        X = np.append(X, np.array([[float(row[0]),float(row[1]),float(row[2]),float(row[3]),float(row[4]),float(row[5])]]), axis=0)
                        y = np.append(y, np.array([int(label)]), axis=0)

    return X, y

def classificationModel(model):
    models = {
        "LR"        : LogisticRegression(max_iter=1000,tol=0.001),
        "KNN"       : KNeighborsClassifier(n_neighbors=5, weights='uniform'),
        "DT"        : DecisionTreeClassifier(),
        "SVM"       : svm.SVC(kernel='linear'),
        'NBayes'    : GaussianNB(),
        'Perceptron': Perceptron(max_iter=10000, eta0=0.00001, random_state=0, alpha=0.01),
        'MLP'       : MLPClassifier(hidden_layer_sizes=(3), activation="relu", solver='adam', 
                                    alpha=0.001, batch_size='auto', learning_rate="constant",
                                    learning_rate_init=0.001, power_t=0.5, max_iter=300 ,tol=1e-4)
    }

    return models.get(model, None)

def readTimeSeriesData(DATA_PATH, INTERVAL, dataType):
    LABELS = ['0','1']
    if dataType == "acc":

        X = np.empty((0,20,3))
        y = np.empty((0))
        
        for label in LABELS:
            csvPath = os.path.join(DATA_PATH,label)
            filenames = glob.glob(csvPath + "/*.csv")

            for filename in filenames:
                print(filename)
                s,e = 0,20
                with open(filename,'r') as csvfile:
                    content = csv.reader(csvfile)
                    rows = np.array([row for row in content])
                    iter = int(len(rows)/INTERVAL)

                    for i in range(0,iter-1):
                        X = np.append(X, rows[np.newaxis,s:e,0:3].astype(np.float), axis=0)
                        y = np.append(y, np.array([int(label)]), axis=0)
                        s += INTERVAL
                        e += INTERVAL

                        # print(str(s) + ' ' + str(e))
    else:

        X = np.empty((0,20,6))
        y = np.empty((0))
        
        for label in LABELS:
            csvPath = os.path.join(DATA_PATH,label)
            filenames = glob.glob(csvPath + "/*.csv")

            for filename in filenames:
                print(filename)
                s,e = 0,20
                with open(filename,'r') as csvfile:
                    content = csv.reader(csvfile)
                    rows = np.array([row for row in content])
                    iter = int(len(rows)/INTERVAL)

                    for i in range(0,iter-1):
                        X = np.append(X, rows[np.newaxis,s:e,0:6].astype(np.float), axis=0)
                        y = np.append(y, np.array([int(label)]), axis=0)
                        s += INTERVAL
                        e += INTERVAL

                        # print(str(s) + ' ' + str(e))
                
    return X, y

def buildModel(inputShape, modelName="LSTM"):

    if modelName == "LSTM":
        model = Sequential()
        model.add(LSTM(10, activation='relu', return_sequences=False, input_shape=inputShape))
        model.add(Dense(1))
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'] )
          
    if modelName == "BD_LSTM":
        model = Sequential()
        model.add(Bidirectional(LSTM(10, activation='tanh', return_sequences=False), input_shape=inputShape))
        model.add(Dense(1))
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])    
    
    model.summary()

    return model      
        