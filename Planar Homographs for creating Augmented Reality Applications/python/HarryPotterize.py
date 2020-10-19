import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts
from matchPics import matchPics
from planarH import computeH_ransac
from helper import plotMatches
from planarH import compositeH
import matplotlib.pyplot as plt
from helper import computeBrief
from helper import corner_detection

#'threshold for corner detection using FAST feature detector'

#Import necessary functions
#Write script for Q2.2.4
opts = get_opts()
'''
Write a script HarryPotterize.py that
1. Reads cv cover.jpg, cv desk.png, and hp cover.jpg.
2. Computes a homography automatically using MatchPics and computeH ransac.
3. Uses the computed homography to warp hp cover.jpg to the dimensions of the cv desk.png image using the OpenCV function cv2.warpPerspective function.
4. At this point you should notice that although the image is being warped to the correct location, it is not filling up the same space as the book. Why do you think this is happening? How would you modify hp cover.jpg to fix this issue?
5. Implement the function:
composite img = compositeH( H2to1, template, img )
to now compose this warped image with the desk image as in in Figure 4
6. Include your result in your write-up.

'''

cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
#print(cv_desk.shape)
hp_cover=cv2.imread('../data/hp_cover.jpg')
#print(hp_cover.shape)

matches1,locs1,locs2=matchPics(cv_desk,cv_cover,  opts)#Match between first two images

locs1[:,[0,1]] = locs1[:,[1,0]]
locs2[:,[0,1]] = locs2[:,[1,0]]

plotMatches(cv_desk,cv_cover,  matches1,locs1,locs2)

print('locs1m1 Shape :',locs1.shape)
print('locs2m1 Shape :',locs2.shape)

print('matches :',matches1.shape)

bestH2to1,inliers=computeH_ransac(locs1[matches1[:,0]],locs2[matches1[:,1]],opts)

#print('Best :',bestH2to1)
dim=(cv_cover.shape[1],cv_cover.shape[0])
hp_cover=cv2.resize(hp_cover,dim)

composite_img=compositeH(bestH2to1,hp_cover ,cv_desk)
print("Shape of composite image:",composite_img.shape)

cv2.imwrite('../data/warp_exp1.jpg',composite_img)




