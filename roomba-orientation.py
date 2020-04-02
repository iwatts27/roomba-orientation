# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 14:25:57 2017

@author: iwatts
"""

import re
import glob
import cv2
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import hog 
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor as MLP

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

t = time.time()

# Load ANN and normalization scalar
ANN = pickle.load(open("ANN.pkl","rb"))
trans = pickle.load(open("normalizationScalar.pkl","rb"))

""" Use this if loading an original image and not 360 rotated images
images = []
# Load image
im = cv2.imread('Roomba02.jpg')
# Get dimensions
width, height, depth = im.shape
# Get rotation centers
x0 = height/2
y0 = width/2
# Get center box edges
left = (width - 200)/2
top = (height - 200)/2
right = (width + 200)/2
bottom = (height + 200)/2

for i in range(360):
    # Get rotational matrix        center  angle scale
    M = cv2.getRotationMatrix2D((x0,y0),  i,    1  ) 
    # Rotate image        input  matrix          output shape
    rot = cv2.warpAffine( im,    M,    (im.shape[1],im.shape[0]))
    # Crop image to 200x200 from center
    rot = rot[int(left):int(right), int(top):int(bottom)]
    # Create file name
    fileName = 'images2/image_' + str(i) + '.png'
    # Save modified image
    cv2.imwrite(fileName,rot)
"""

# Load images
files = sorted(glob.glob('test/*.jpg'), key=natural_key)
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
transFeat = trans.fit_transform(feat)
# Make angle predictions using the ANN
N = ANN.predict(transFeat)
# Convert prediction to angle in degrees
degN = np.rad2deg(np.arctan2(N[:,0],N[:,1]))
# Convert negative angles
degN[degN < 0] = degN[degN < 0] + 360
# Calculate absolute erorrs in degrees
AE = []
AE = np.rad2deg(np.abs(np.arctan2(labels[:,0],labels[:,1]) - np.arctan2(N[:,0],N[:,1])))
# Convert errors > 180
AE[AE > 180] = 360 - AE[AE > 180]
# Calculate max and mean error
maxAE = np.max(AE)
MAE = np.mean(AE)

print('Mean Absolute Error in Degrees:',MAE)
print('Max Absolute Error in Degrees:',maxAE)

# Graph error vs ground truth state
plt.figure(3)
plt.clf()
plt.plot(range(len(files)),AE)

# Graph matrix of roombas with prediction arrows
f, ax = plt.subplots(15,24)
k = 0
for i in range(15):
    for j in range(24):
        img = mpimg.imread(files[k])
        ax[i][j].imshow(img)
        ax[i][j].arrow(100, 100, N[k*-1,1]*50,N[k*-1,0]*50, fc="c", ec='m', head_width=10, head_length=15, linewidth=3)
        ax[i][j].get_xaxis().set_visible(False)
        ax[i][j].get_yaxis().set_visible(False)
        k += 1

t2 = time.time()
print('Run time (s):', round(t2-t, 2))
