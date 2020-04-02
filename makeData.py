# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 11:47:39 2017

@author: iwatts
"""

import cv2

# Load images
im1 = cv2.imread('Roomba01.jpg')
im2 = cv2.imread('Roomba02.jpg')
# Get dimensions
width1, height1, depth1 = im1.shape
width2, height2, depth2 = im2.shape
# Get rotation centers
x01 = height1/2
y01 = width1/2
x02 = height2/2
y02 = width2/2
# Get center box edges
left1 = (width1 - 200)/2
top1 = (height1 - 200)/2
right1 = (width1 + 200)/2
bottom1 = (height1 + 200)/2
left2 = (width2 - 200)/2
top2 = (height2 - 200)/2
right2 = (width2 + 200)/2
bottom2 = (height2 + 200)/2

for i in range(360):
    # Get rotational matrix        center  angle scale
    M1 = cv2.getRotationMatrix2D((x01,y01),  i,    1  ) 
    M2 = cv2.getRotationMatrix2D((x02,y02),  i,    1  ) 
    # Rotate image        input  matrix          output shape
    rot1 = cv2.warpAffine( im1,    M1,    (im1.shape[1],im1.shape[0]))
    rot2 = cv2.warpAffine( im2,    M2,    (im2.shape[1],im2.shape[0]))
    # Crop image to 200x200 from center
    rot1 = rot1[int(left1):int(right1), int(top1):int(bottom1)]
    rot2 = rot2[int(left2):int(right2), int(top2):int(bottom2)]
    # Create file name
    fileName1 = 'images/image1_' + str(i) + '.png'
    fileName2 = 'images/image2_' + str(i) + '.png'
    # Save modified image
    cv2.imwrite(fileName1,rot1)
    cv2.imwrite(fileName2,rot2)

    