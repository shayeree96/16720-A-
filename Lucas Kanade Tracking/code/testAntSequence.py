import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import SubtractDominantMotion
import time

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--tolerance', type=float, default=0.75, help='binary threshold of intensity difference when computing the mask')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance

seq = np.load('../data/antseq.npy')
imH,imW,frames = np.shape(seq)
# rect = [101., 61., 155., 107.]
start=time.time()
for i in range(frames-1):
    #print(i)
    image1 = seq[:,:,i]
    image2 = seq[:,:,i+1]
    mask = SubtractDominantMotion.SubtractDominantMotion(image1,image2,threshold, num_iters, tolerance)
    
    if (i == 29) or (i == 59) or (i == 89) or (i ==119):
        # since we are plotting framee 30, 60, 90, 120 are plotted
        pic = plt.figure()
        plt.imshow(image2, cmap='gray')
        plt.axis('off')
        plt.title("Frame  %d "%(i+1))
        # print(mask)
        for w in range(mask.shape[0]-1):
            for h in range(mask.shape[1]-1):
                if mask[w,h]:
                    plt.scatter(h, w, s = 1, c = 'r', alpha=0.5)
        plt.show()
        
        
        
stop=time.time()
print("Total time taken:",stop-start)   