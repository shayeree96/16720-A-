
import numpy as np
import cv2 

def loadVid(path):
	# Create a VideoCapture object and read from input file
	# If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(path)
	 
	# Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    i = 0
	# Read until video is completed
    while(cap.isOpened()):
		# Capture frame-by-frame]
        i += 1
        ret, frame = cap.read()
        #print('Shape of frame captured:',frame.shape)
        if ret == True:

			#Store the resulting frame
            if i == 1:
                frames = frame[np.newaxis, ...]
                #print('in i==1:',frames.shape)
            else:
                frame = frame[np.newaxis, ...]
                #print('Frame new axis in else:',frames.shape)
                frames = np.vstack([frames, frame])
                #print('Frame new axis in else:',frames.shape)
                frames = np.squeeze(frames)
			
        else:
            break
	 
	# When everything done, release the video capture object
    cap.release()
    
    return frames
#k=loadVid('../data/ar_source.mov')
    
