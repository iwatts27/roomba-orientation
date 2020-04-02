# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 14:33:59 2017

@author: iwatts
"""

import re
import glob
import cv2
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from skimage.feature import hog 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as split
from sklearn.neural_network import MLPRegressor as MLP

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

t = time.time()

# Load images
files = sorted(glob.glob('images/*.png'), key=natural_key)
feat  = np.empty((len(files),9720))
labels = np.zeros((len(files),2))
j=0
# Loop through roomba images and create feature data
for i in files:
    # Read image
    im = cv2.imread(i)
    # Determine ground truth state
    angle = int(i[i.index('_')+1:i.index('.')])
    # Create labels
    labels[j][0] = np.sin(np.deg2rad(angle))
    labels[j][1] = np.cos(np.deg2rad(angle))
    # Convert image to HSV colorspace
    imHSV = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    # Convert image to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # Compute HOG feature data
    hogFeat = hog(gray, orientations=30, pixels_per_cell=(20, 20), cells_per_block=(2, 2))
    # Compute color histogram data  Removing the color features increased performance
    #colorFeatH = cv2.calcHist([imHSV], [0], None, [179], [0,179], False)
    #colorFeatS = cv2.calcHist([imHSV], [1], None, [256], [0,256], False)
    #colorFeatV = cv2.calcHist([imHSV], [2], None, [256], [0,256], False)
    # Merge feature data`
    #featInter = np.hstack((hogFeat,colorFeatH.reshape(len(colorFeatH))
    #                              ,colorFeatS.reshape(len(colorFeatS))
    #                              ,colorFeatV.reshape(len(colorFeatV))))
    # Add feature data to main feature array
    feat[j][:] = hogFeat 
    j+=1
# Normalize feature data
trans = StandardScaler()
featTrans = trans.fit_transform(feat)
# Split test and training data
featTrain, featVal, labelsTrain, labelsVal = split(
        featTrans, labels, test_size=0.20)
#featTrain = trans.fit_transform(featTrain)
#featVal = trans.fit_transform(featVal)

pickle.dump(featTrain,open("featTrain.pkl","wb"))
pickle.dump(featVal,open("featVal.pkl","wb"))
pickle.dump(labelsTrain,open("labelsTrain.pkl","wb"))
pickle.dump(labelsVal,open("labelsVal.pkl","wb"))
"""
featTrain = pickle.load(open("featTrain.pkl","rb"))
featVal = pickle.load(open("featVal.pkl","rb"))
labelsTrain = pickle.load(open("labelsTrain.pkl","rb"))
labelsVal = pickle.load(open("labelsVal.pkl","rb"))
"""

# Determine optimal number of hidden nodes
print(1)
runs = 10
nodes = 14
lowestAvgMAE = float('inf')
runPoints = []
avgPoints = []
for i in range(2,nodes+2):
    errorSum = []
    for j in range(runs):
        ANN = MLP(hidden_layer_sizes = (i,), max_iter=200,
              activation='logistic',solver='lbfgs')
        ANN.fit(featTrain,labelsTrain)
        N = ANN.predict(featVal)
        k = len(labelsVal)
        labelDeg = np.rad2deg(np.arctan2(labelsVal[:,0],labelsVal[:,1]))
        NDeg = np.rad2deg(np.arctan2(N[:,0],N[:,1]))
        labelDeg[labelDeg < 0] = labelDeg[labelDeg < 0] + 360 
        NDeg[NDeg < 0] = NDeg[NDeg < 0] + 360 
        AE = np.abs(labelDeg - NDeg)
        AE[AE > 180] = 360 - AE[AE > 180]
        MAE = np.mean(AE)
        runPoints.append([i,MAE])
        errorSum.append(MAE)
    avgMAE = np.mean(errorSum)
    avgPoints.append([i,avgMAE])
    if avgMAE < lowestAvgMAE:
        lowestAvgMAE = avgMAE
        bestH = i

# Graph performance by number of hidden nodes
plt.figure(1,figsize=(12,4),dpi=100)
plt.clf()
plt.grid()
for i in range(len(runPoints)):
    plt.scatter(x = runPoints[i][0], y = runPoints [i][1], facecolors = 'none', color = 'r')
for i in range(len(avgPoints)):
    plt.scatter(x = avgPoints[i][0], y = avgPoints [i][1],marker = '*', s=100, color = 'm')
plt.xlabel('Number of hidden nodes')
plt.ylabel('MAE(degrees)')
plt.savefig('figure3.png')

# Train ANN
ANN = MLP(hidden_layer_sizes = (bestH,), activation='logistic', max_iter=1,
                      solver='lbfgs', warm_start=True)
epochs = []
MAEtrain = []
MAEval = []
previousMAE = float('inf')

print(2)
for i in range(1000):
    ANN.fit(featTrain,labelsTrain)
    epochs.append(ANN.n_iter_)
    yPredict = ANN.predict(featTrain)
    MAEtrain.append(np.mean(np.abs(yPredict - labelsTrain)))
    yPredict = ANN.predict(featVal)
    MAEval.append(np.mean(np.abs(yPredict - labelsVal)))
    if MAEval[-1] >= previousMAE and ANN.n_iter_ % 10 == 0:
        break
    if ANN.n_iter_ % 10 == 0:
        previousMAE = MAEval[-1]

# Graph training and validation error vs training time
plt.figure(2,figsize=(12,4),dpi=100)
plt.clf()
plt.semilogy(epochs,MAEval,label='validation')
plt.semilogy(epochs,MAEtrain,label='training')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('MAE in ANN output')
plt.savefig('figure2.png')

# Pickle ANN for later use
pickle.dump(ANN,open("ANN.pkl","wb"))
pickle.dump(trans,open("normalizationScalar.pkl","wb"))

t2 = time.time()
print('Run time (s):', round(t2-t, 2))