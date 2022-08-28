# Author: Pinyuan Feng
# Created on Dec. 1st, 2021

import os
import matplotlib.pyplot as plt # plt 用于显示图片
import mpl_toolkits.mplot3d.axes3d as axes3d
import matplotlib.animation as animation
import numpy as np
import glob
import csv
from random import shuffle

DATA_PATH = '../data/'
LABELS = ['0','1']
DATA_TYPE = "accGyro"

data1 = []
data0 = []
for label in LABELS:
    csvPath = os.path.join(DATA_PATH,label)
    # print(csvPath)
    filenames = glob.glob(csvPath + "/*.csv")
    for filename in filenames:
        with open(filename,'r') as csvfile:
            content = csv.reader(csvfile)
            if DATA_TYPE == "acc":
                rows = [[float(row[0]),float(row[1]),float(row[2])] for row in content]
            else:
                rows = [[float(row[3]),float(row[4]),float(row[5])] for row in content]
            
            shuffle(rows)

            if label == LABELS[0]:
                data0.extend(rows)
            if label == LABELS[1]:
                data1.extend(rows)

data1 = np.array(data1) 
data0 = np.array(data0)
# print(data0.shape)
# print(data1.shape)
# print(data0[0:5])
# print(data0[0:5,0])
       
# Attaching 3D axis to the figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('3D Data Visualization')

# Setting the axes properties
if DATA_TYPE == "acc":
    ax.set_xlim3d([0.0, 0.3])
    ax.set_xlabel('Delat X')
    ax.set_ylim3d([0.0, 0.3])
    ax.set_ylabel('Delat Y')
    ax.set_zlim3d([0.0, 0.3])
    ax.set_zlabel('Delat Z')
else:
    ax.set_xlim3d([0.0, 0.1])
    ax.set_xlabel('Angular velocity X')
    ax.set_ylim3d([0.0, 0.1])
    ax.set_ylabel('Angular velocity Y')
    ax.set_zlim3d([0.0, 0.1])
    ax.set_zlabel('Angular velocity Z')

# Plotting data
ax.scatter(data0[0:1000,0], data0[0:1000,1], data0[0:1000,2], c="g",marker=".")
ax.scatter(data1[0:1000,0], data1[0:1000,1], data1[0:1000,2], c="r",marker="+")

plt.show()
