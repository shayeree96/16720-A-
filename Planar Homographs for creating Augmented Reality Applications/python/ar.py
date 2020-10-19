#Import necessary functions
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
    
#print("Shape of pad :",pad.shape)
ar=np.concatenate((ar,pad),axis=0)    
    
#print("Shape of ar after padding:",ar.shape)
#Write script for Q3.1

opts = get_opts()
locs1_arr=[]
locs2_arr=[]
matches_arr=[]
bestH2to1_arr=[]
cv_cover = cv2.imread('../data/cv_cover.jpg')
composite_list=[]

for i in range(book.shape[0]):
    print('i :{} and book :{}'.format(i,book[i,:,:,:].shape))
    matches,locs1,locs2=matchPics(book[i,:,:,:],cv_cover,opts)
    locs1[:,[0,1]] = locs1[:,[1,0]]
    locs2[:,[0,1]] = locs2[:,[1,0]]
    #plotMatches(book[i,:,:,:],cv_cover,matches,locs1,locs2)
    bestH2to1,inliers=computeH_ransac(locs1[matches[:,0]],locs2[matches[:,1]],opts)
    dim=(cv_cover.shape[1],cv_cover.shape[0])
  
    aspect_ratio=cv_cover.shape[1]/cv_cover.shape[0]#w/h
    #print('Aspect ratio :',aspect_ratio)
    
    video_cover=ar[i,:,:,:]
    video_cover=video_cover[44:-44,:]#We chop off the black portions

    H,W,C=video_cover.shape
    #print('Shape of ar:',video_cover_cropped.shape)
    h=H/2
    w=W/2
    width_ar=H*cv_cover.shape[1]/cv_cover.shape[0]
    video_cover=video_cover[:,int(w-width_ar/2):int(w+width_ar/2)]#Height is fixed
    video_cover=cv2.resize(video_cover,dim)
    
    composite_img=compositeH(bestH2to1,video_cover,book[i,:,:,:])
    composite_list.append(composite_img)  

       

def make_video(images, outimg=None, fps=5, size=None,
               is_color=True, format="XVID",outvid="../data/output_3.mov"):
    
    from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
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
make_video(composite_list,fps=25)
