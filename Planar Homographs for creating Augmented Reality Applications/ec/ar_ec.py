#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 13:25:44 2020

@author: shayereesarkar
"""

import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts
from matchPics import matchPics
from planarH import computeH_ransac
from helper import plotMatches
from planarH import compositeH
from loadVid import loadVid
from PIL import Image
import skimage.feature
import matplotlib.pyplot as plt
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
#import matplotlib.pyplot as plt

ar_video_path='../data/ar_source.mov'
book_vid_path='../data/book.mov'

#We load the video
ar=loadVid(ar_video_path)
book=loadVid(book_vid_path)
extend=book.shape[0]-ar.shape[0]

#1. You can pad the shorter video by looping the initial frames.
pad=np.zeros((extend,ar.shape[1],ar.shape[2],ar.shape[3]))
for i in range(extend):
    pad[i,:,:,:]=ar[i,:,:,:]
    
ar=np.concatenate((ar,pad),axis=0)    

#Write script for Q3.1
opts = get_opts()
ratio = opts.ratio 

cv_cover = cv2.imread('../data/cv_cover.jpg')
composite_list=[]

#We will use ORB() detector
orb=cv2.ORB_create(nlevels=4)

for i in range(book.shape[0]):
    print('i :{} and book :{}'.format(i,book[i,:,:,:].shape))
    
    # find keypoints compute the descriptors with ORB
    locs1, desc1 = orb.detectAndCompute(book[i,:,:,:], None)
    locs2, desc2 = orb.detectAndCompute(cv_cover, None)
    
    img1 = cv2.drawKeypoints(book[i,:,:,:], locs1, None, color=(0,255,0), flags=0)
    plt.imshow(img1), plt.show()
    
    #Match the Features
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(desc1, desc2, None)
    
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    
    for i, match in enumerate(matches):
        points1[i, :] = locs1[match.queryIdx].pt
        points2[i, :] = locs2[match.trainIdx].pt
      
      # Find homography
    bestH2to1, mask = cv2.findHomography(points1, points2, cv2.RANSAC,2.0)
    
    #print(bestH2to1)
    dim=(cv_cover.shape[1],cv_cover.shape[0])
  
    aspect_ratio=cv_cover.shape[1]/cv_cover.shape[0]#w/h
    #print('Aspect ratio :',aspect_ratio)
    
    video_cover=ar[i,:,:,:]
    video_cover=video_cover[44:-44,:]#We chop off the black portions

    H,W,C=video_cover.shape
    
    h=H/2
    w=W/2
    width_ar=H*cv_cover.shape[1]/cv_cover.shape[0]
    
    video_cover=video_cover[:,int(w-width_ar/2):int(w+width_ar/2)]#Height is fixed
    video_cover=cv2.resize(video_cover,dim)
    
    composite_img=compositeH(bestH2to1,video_cover,book[i,:,:,:])
    composite_list.append(composite_img)  
    

def make_video(images, outimg=None, fps=10, size=None,
               is_color=True, format="XVID",outvid="../data/output_3.mov"):

    fourcc = VideoWriter_fourcc(*format)
    vid = None
    for img in images:
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
        vid.write(np.uint8(img))
    vid.release()
    return vid
make_video(composite_list,fps=10)
