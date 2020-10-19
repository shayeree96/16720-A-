import numpy as np
import cv2
from matchPics import matchPics
import scipy
from opts import get_opts
import matplotlib.pyplot as plt
from helper import plotMatches

opts = get_opts()

#Q2.1.6
#Read the image and convert to grayscale, if necessary
scipy.ndimage.rotate
cv_cover = cv2.imread('../data/cv_cover.jpg')

hist=[]
degree=[]
for i in range(1,36):
	#Rotate Image
    deg=i*10
    degree.append(deg)
    img_rot=scipy.ndimage.rotate(cv_cover,deg)
    #angle.appen(deg)
    matches,locs1,locs2=matchPics(cv_cover, img_rot, opts)
	
	#Compute features, descriptors and Match features
    plotMatches(cv_cover, img_rot, matches, locs1, locs2)

	#Update histogram
    hist.append(matches.shape[0])

print(hist)

bin = np.linspace(10,350,num = 35,endpoint=True)
plt.bar(bin, hist[1:],width=bin[1] - bin[0])
plt.xlim(min(bin)-5, max(bin)+5)
plt.show()


hist_new=[0]*36
bin = np.linspace(10,350,num = 35,endpoint=True)
plt.bar(bin, hist[1:],width=bin[1] - bin[0])
plt.xlim(min(bin)-5, max(bin)+5)
plt.show()

#Display histogram

