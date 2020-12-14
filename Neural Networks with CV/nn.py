import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

############################## Q 2.1 ##############################
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    
    
    upper_bound=np.sqrt(6/(in_size+out_size))
    lower_bound=-upper_bound
    W=np.random.uniform(lower_bound,upper_bound,(in_size,out_size))
    b=np.zeros((out_size))

    params['W' + name] = W
    params['b' + name] = b
    
############################## Q 2.2.1 ##############################
# x is a matrix
# a sigmoid activation function
def sigmoid(x): #Remember to apply the log sum trick
    res = 1/(1+np.exp(-x))
    return res

############################## Q 2.2.1 ##############################
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]
    
    pre_act= (X @ W)+b
    post_act=activation(pre_act)
    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

############################## Q 2.2.2  ##############################
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    
    max_xi = np.max(x, axis=1)
    shift_x = x - np.expand_dims(max_xi, axis=1)
    res = np.exp(shift_x)
    res = res/np.expand_dims(np.sum(res, axis=1), axis=1)
    
    return res


############################## Q 2.2.3 ##############################
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    
    acc = 0
    loss=-np.sum(y*np.log(probs))
    
    for i in range(y.shape[0]):
        if(np.argmax(y[i,:])==np.argmax(probs[i,:])):
            acc+=1
    acc=acc/y.shape[0]

    return loss, acc 

############################## Q 2.3 ##############################
# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]

    # do the derivative through activation first
    
    grad_output= activation_deriv(post_act)*delta
    grad_W = X.T @ grad_output
    grad_X = grad_output @ W.T
    grad_b = (np.ones((1,X.shape[0]))@ grad_output).flatten()

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    
    return grad_X

############################## Q 2.4 ##############################
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches=[]
    
    random_batches = np.array_split(np.random.permutation(x.shape[0]), x.shape[0]//batch_size)
    
    for i in random_batches:
         batches.append((x[i,:],y[i,:]))
        
    return batches

