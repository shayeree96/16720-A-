import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import *
from scipy.interpolate import RectBivariateSpline
from TemplateCorrection import *
# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--template_threshold', type=float, default=5, help='threshold for determining whether to update template')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
template_threshold = args.template_threshold

seq = np.load("../data/carseq.npy")
rect = [59, 116, 145, 151]
rect_list=[]
rect_old=np.load("../result/carseqrects.npy")
#print('Shape of rect_old',rect_old.shape)

width=rect[2]-rect[0]
height=rect[3]-rect[1]

#For the first frame we get the template spline evaluation and keep subtracting it

It=seq[:,:,0]
x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]#2.
row,column=It.shape#3.
row_rec=x2-x1
col_rec=y2-y1
y=np.arange(0,row,1)#6.
x=np.arange(0,column,1)#7.
cc,rr=np.meshgrid(np.linspace(x1,x2,col_rec),np.linspace(y1,y2,row_rec))

splinet=RectBivariateSpline(y,x,It)
T=splinet.ev(rr,cc)

#We add tbe template correction
for i in range(1,seq.shape[2]-1):
    
    #print("Processing frame %d" % i)
    #We have to send each frame to lucas Kanade
    It=seq[:,:,i-1]
    It1=seq[:,:,i]
    rect_list.append(rect)
    p=LucasKanade(It, It1, rect,threshold,num_iters, p0 = np.zeros(2))
    rect[0]+=p[0]#x1
    rect[1]+=p[1]#y1
    rect[2]+=p[0]#x2
    rect[3]+=p[1]#y2
    
    #now we resend these coordinates to Lucas Kanade for template collection
       
    p_new=TemplateCorrection(T, It1, rect,threshold,num_iters, p0 = np.zeros(2))
    
    if (np.linalg.norm(p_new-p))<template_threshold:
        rect[0]+=p_new[0]#x1
        rect[1]+=p_new[1]#y1
        rect[2]+=p_new[0]#x2
        rect[3]+=p_new[1]#y2
       
    #For template correction we first send the original images first frame 
    
    if i%100==0 or i==1:
        #We will print the patches
        plt.figure()
        plt.imshow(seq[:,:,i],cmap='gray')
        rectangle1=patches.Rectangle((int(rect[0]),int(rect[1])),width,height,fill=False, edgecolor='r', linewidth=2)
        rectangle2=patches.Rectangle((int(rect_old[i][0]),int(rect_old[i][1])),width,height,fill=False, edgecolor='b', linewidth=2)
        
        plt.gca().add_patch(rectangle1)
        plt.gca().add_patch(rectangle2)
        plt.title('frame %d'%i)
        plt.show()
        
np.save('../result/carseqrects-wcrt.npy',rect_list)        
        
    
    
    
    
    
    
    
    