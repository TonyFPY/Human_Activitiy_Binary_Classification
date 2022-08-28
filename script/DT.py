# Author: Pinyuan Feng
# Created on Dec. 1st, 2021
# Decision Tree
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html?highlight=decision%20tree#sklearn.tree.DecisionTreeClassifier

import os
import numpy as np
import glob
import csv
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support as score

DATA_PATH = '../data/'
LABELS = ['0','1']

X = np.empty((0,3))
y = np.empty((0,1))
for label in LABELS:
    csvPath = os.path.join(DATA_PATH,label)
    filenames = glob.glob(csvPath + "/*.csv")
    for filename in filenames:
        # print(filename)
        with open(filename,'r') as csvfile:
            content = csv.reader(csvfile)
            for row in content:
                X = np.append(X, np.array([[float(row[0]),float(row[1]),float(row[2])]]), axis=0)
                y = np.append(y, np.array([[int(float(row[3]))]]), axis=0)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2) 

# Data Normalization
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Train
model = DecisionTreeClassifier()
model.fit(X_train_std,y_train)

# Test
start = time.perf_counter()

precision, recall, fscore = 0, 0, 0
for i in range(0,10):
    y_predict = model.predict(X_test_std)
    p, r, f, _ = score(y_test, y_predict, average='weighted', warn_for=('precision', 'recall', 'f-score'), sample_weight=None, zero_division=0)
    precision += p
    recall += r 
    fscore += f
precision = precision/10
recall = recall/10
fscore = fscore/10

end = time.perf_counter()

print('F1-score : {:.2%}'.format(fscore))
print('Precision: {:.2%}'.format(precision))
print('Recall   : {:.2%}'.format(recall))
print('Running time: %s seconds'%(end-start))

