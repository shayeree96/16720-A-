import os, multiprocessing
from os.path import join, isfile
import sklearn.cluster 
import numpy as np
from PIL import Image
import scipy.ndimage
import skimage.color
from numpy.random import default_rng
import matplotlib.pyplot as plt

def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    filter_scales = opts.filter_scales
    # ----- TODO -----
     
    #print('filter_scales :',filter_scales)
    if len(img.shape)>=3:
        H,W,C=img.shape
    else:
        H,W=img.shape
        C=1
        
    Filter_size=4

    #2.Make sure if the channels are 3 or not and if less we take care of the grayscale images
    if C==1: # grey 
        img=np.expand_dims(img, axis=2)
        #print("Dimension Expansion :",img.shape)
        img = np.tile(img,(1,1,3))
        H,W,C=img.shape
        #print("in filtering :",img.shape)
    if C == 4: # special case
        img = img[:,:,0:3]
        H,W,C=img.shape
        
    #1.Make sure to noramlize the image first by checking that entries in image are float and with range 0 1
    if (type(img[0,0,0])==int): 
        img = img.astype('float') / 255
    elif (np.amax(img) > 1.0):
        img = img.astype('float') / 255    
           
    #3.We convert the image to skimage.color.rgb2lab()  
    img=skimage.color.rgb2lab(img)
      
    #now the shape is (H,W,3F*2) cause of the 2 filter scales that we have
    filter_responses=np.zeros((H,W,C*len(filter_scales)*Filter_size)) 
    #print('Original shape :',img.shape)
    #print('filter_responses shape :',filter_responses.shape)
    #4.We apply padding across the edges of the filter responses carefully
    
    c=0       
    
    for i in range(img.shape[2]):#Looping through each channel
        for j in range(len(filter_scales)):#For each of the filter scales we have to calculate this 
            filter_responses[:,:,c]=scipy.ndimage.gaussian_filter(img[:,:,i], sigma=filter_scales[j])
            filter_responses[:,:,c+3]=scipy.ndimage.gaussian_laplace(img[:,:,i], sigma=filter_scales[j])
            filter_responses[:,:,c+6]=scipy.ndimage.filters.gaussian_filter(img[:,:,i], sigma=filter_scales[j],order=[0,1])  
            filter_responses[:,:,c+9]=scipy.ndimage.filters.gaussian_filter(img[:,:,i], sigma=filter_scales[j],order=[1,0])
            c=c+C*Filter_size
        c=i+1
    return filter_responses

def compute_dictionary_one_image(*args):
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    '''

    # ----- TODO -----
    
    #Here we pass the image 
    
    #The arguments will be the image path and the alpha from which we want to extract the responses
    path,opts=args
    image=skimage.io.imread(path)# We get the nd array for the image
    #print("Shape of image :",image.shape)
    alpha=opts.alpha
    
    #Now we pass the image to extract filter responses to get the response
    response=extract_filter_responses(opts, image)  
    #print("Shape of response :",response.shape)
    
    #Now after getting the responses we sample alpha pixels in HXW from them
    H,W,C=response.shape
    response_final=np.reshape(response,(H*W,C))
    rng = default_rng()
    slices = rng.choice(H*W, size=alpha, replace=False)
    #slices=np.unique((H*W),size=alpha)
    #print('After reshape :',response.shape)
    response_final=response_final[slices,:]#we take alpha responses after flattening it
    
    #now we save the response
    #We ge the filter responses and now we need to extract random alpha of these responses
    
    return response_final

def compute_dictionary(opts, n_worker=4):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel
    
    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K

    train_files = open(join(opts.data_dir, 'train_files.txt')).read().splitlines()
    # ----- TODO -----
    #now we have to take this image response continuously
    
    print("In compute dicitonary")
    img_response=compute_dictionary_one_image(os.path.join(opts.data_dir,(train_files[0])),opts)
    img_stack=np.zeros((0,img_response.shape[1]))
    for i in range(len(train_files)):
        img_response=compute_dictionary_one_image(os.path.join(opts.data_dir,(train_files[i])),opts)
        img_stack=np.vstack((img_stack,img_response))
        print(img_stack.shape)
        
    kmeans=sklearn.cluster.KMeans(n_clusters=K,n_jobs=-1).fit(img_stack)
    dictionary = kmeans.cluster_centers_
    np.save('dictionary.npy',dictionary)

    return dictionary
         
    

    ## example code snippet to save the dictionary
    # np.save(join(out_dir, 'dictionary.npy'), dictionary)

def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    
    filter_response=extract_filter_responses(opts,img)
    
    
    H,W,C=filter_response.shape
    K,C_same=dictionary.shape
    wordmap=np.zeros((H,W))
    closeness = np.zeros(K)
    #print(img)
    for i in range(H):
        for j in range(W):
            pixel=filter_response[i][j][:]#Get that pixel vector
            pixel=np.reshape(pixel,(1,C_same))    
            #now we find the euclidean distance for this wrt to the 10 clusters I have in the dictionary
            closeness=scipy.spatial.distance.cdist(dictionary,pixel)
            #print(closeness.shape)
            wordmap[i,j]=np.argmin(closeness,axis=0)
    #print("Shape is :",wordmap.shape)        
            
    return wordmap
