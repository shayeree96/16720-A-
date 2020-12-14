import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import *
import time

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold

seq = np.load("../data/carseq.npy")
rect = [59, 116, 145, 151]
rect_list=np.zeros((seq.shape[2]-1,4))

for i in range(seq.shape[2]-1):
    
    #We have to send each frame to lucas Kanade
    It=seq[:,:,i]
    It1=seq[:,:,i+1]
    
    p=LucasKanade(It, It1, rect,threshold,num_iters, p0 = np.zeros(2))
    rect[0]+=p[0]#x1
    rect[1]+=p[1]#y1
    rect[2]+=p[0]#x2
    rect[3]+=p[1]#y2
    #print(rect)
    rect_list[i]=rect
    #print("Rec list :",rect_list[i])
    if i%100==0 or i==1:
        #We will print the patches
        width=rect[2]-rect[0]
        height=rect[3]-rect[1]
        
        plt.figure()
        plt.imshow(seq[:,:,i],cmap='gray')
        rectangle=patches.Rectangle((int(rect[0]),int(rect[1])),width,height,fill=False, edgecolor='r', linewidth=2)
        
        plt.gca().add_patch(rectangle)
        plt.title('frame %d'%i)
        plt.show()
        
np.save('../result/carseqrects.npy',rect_list)        
        

