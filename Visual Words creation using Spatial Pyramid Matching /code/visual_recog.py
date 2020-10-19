import os, math, multiprocessing
from os.path import join
from copy import copy

import numpy as np
from PIL import Image

import visual_words


def get_feature_from_wordmap(opts, wordmap):
    '''
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''
    K = opts.K
    H,W=wordmap.shape
    wordmap=np.reshape(wordmap,(1,H*W))
    #No of bins is the no of K rpresent
    bins= np.linspace(0,K,K+1,endpoint = True)
    hist,bin_edges=np.histogram(wordmap,bins,density=True)
    hist=np.reshape(hist,(1,K))
    
    #print(np.sum(hist))
    return hist
   
    # ----- TODO -----
   

def get_feature_from_wordmap_SPM(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3)
    '''
        
    K = opts.K
    L = opts.L
    # ----- TODO -----
    #1.Check the no of layers that are present
    #2.For each layer each channel has to be considered
    #3.Layer 0 and 1 are assigned weights 2^-Lrand others 2^l-L-1
    #4.The layer l for a channel contains 4^l cells in it
    #5.Each cell has a histogram for each channel, each layer
    #6.The final image features are found by adding the histograms after normalization by the total no of features in the image
    
    H,W=wordmap.shape
    
    #We divide this map into 3 layers we want
    #We chop the image into 2^l x 2^l cells in each layer depending on the layer no
    '''
    for i in range(L):#we loop over thr layers
        #Now we can find the no of cells in that layer and calculate histograms for each cell and multiply them by the weight once all are computed
        cells=4**i
        if L==0 or L==1:
            get_feature_from_wordmap(opts,wordmap[])#Send that portion to the wordmap for it to be calculated
            
        else:
    #We get the L1 normalized histogram for this level
    '''
    weight=2**(-L+1)#For layers except 0 or 1
    hist_all=np.zeros((0,K))
    #Now this is the stack for the most fine layer
    layer_num=L+1
    for l in range(layer_num):
        if l==1 or l==0:
            weight=2**(-L)
        else:
            weight=2**(l-layer_num)
        for i in range(0,2**l):
            for j in range(0,2**l):
                index1=int(H/(2**L))
                index2=int(W/(2**L))
                histogram=get_feature_from_wordmap(opts,wordmap[index1*i:index1*(i+1),index2*j:index2*(j+1)])
                #now we weight the histogram
                #Depends on the layer
                hist_all=np.vstack((hist_all,histogram*weight))
    #now we normalize these total weights present   
    h,w=hist_all.shape        
    hist_all=(hist_all/np.sum(hist_all))# Not sure about this step of normalizing it
    hist_all=np.reshape(hist_all,(1,h*w)) 
    
    return hist_all       

    
def get_image_feature(opts, img_path, dictionary):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K)
    '''
    
    # ----- TODO -----
    img = Image.open(img_path)
    #print('image path :',img_path)
    img = np.array(img).astype(np.float32)/255
    wordmap = visual_words.get_visual_words(opts, img, dictionary)
    
    return get_feature_from_wordmap_SPM(opts, wordmap)
    
    

def build_recognition_system(opts, n_worker=1):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    dictionary = np.load(join(out_dir, 'dictionary.npy'))

    # ----- TODO -----
    K=opts.K
    L=opts.L
    #print("L is:",L)
    hist_features=np.zeros((0,int(K*(4**(L+1)-1)/3)))
    #print(hist_features.shape)
    labels=[]
    
    for i in range(0,len(train_files)):
        img_path=os.path.join(data_dir,train_files[i])
        hist_all=get_image_feature(opts, img_path, dictionary)#for that image
        hist_features=np.vstack((hist_features,hist_all))
        labels.append(train_labels[i])
        print(hist_features.shape)
        
    ## example code snippet to save the learned system
    np.savez_compressed(join(out_dir, 'trained_system_2.npz'),
        features=hist_features,
        labels=train_labels,
        dictionary=dictionary,
        SPM_layer_num=opts.L)
    
    
def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''

    # ----- TODO -----
    #We find the distance between the word hist from SPM and histograms
    intersection = np.minimum(histograms,word_hist)
    similarity = np.sum(intersection,axis = 1)
	
    return similarity
    
    
def evaluate_recognition_system(opts, n_worker=1):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, 'trained_system_2.npz'))
    dictionary = trained_system['dictionary']
    trained_features = trained_system['features']
    trained_labels = trained_system['labels']

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system['SPM_layer_num']

    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)
    accuracy=0
    confusion=np.zeros((8,8))
    
    for i in range(0,len(test_files)):
        img_path=os.path.join(data_dir,test_files[i])
        hist_all=get_image_feature(opts, img_path, dictionary)#for that image
        similarity=distance_to_set(hist_all,trained_features)
        prediction_idx=np.argmax(similarity)
        #This will give the label which we can compare to compute accuracy
        predict_label=trained_labels[prediction_idx]
        
        confusion[test_labels[i],predict_label] += 1
        #print('Index: {}. Predict label is:{} and test label :{}'.format(i,predict_label,test_labels[i]))
        #print('Confusion Matrix :',confusion)
    
        accuracy = np.trace(confusion)/np.sum(confusion)#Traces and counts the diagonals in the matrix
        print("{} --> Accuracy:{}".format(i,accuracy))
        
    return(confusion,accuracy)
        


    # ----- TODO -----

