import numpy as np
import scipy.io
from nn import *
from collections import Counter
import matplotlib.pyplot as plt

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# Q5.1 & Q5.2
# initialize layers here
initialize_weights(train_x.shape[1],hidden_size,params,'layer1')
initialize_weights(hidden_size,hidden_size,params,'layer2')
initialize_weights(hidden_size,hidden_size,params,'layer3')
initialize_weights(hidden_size,train_x.shape[1],params,'output')


keys = [key for key in params.keys()]
for k in keys:
    params['m_'+k] = np.zeros(params[k].shape)
train_loss=[]

# should look like your previous training loops
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions

        h1 = forward(xb, params, 'layer1', relu)
        h2 = forward(h1, params, 'layer2', relu)
        h3 = forward(h2, params, 'layer3', relu)
        probs = forward(h3, params, 'output', sigmoid)
        # Loss and accuracy
        loss = np.sum((xb - probs)**2)
        total_loss += loss

        # Backward pass
        delta = 2*(probs-xb)
        delta = backwards(delta, params, 'output', sigmoid_deriv)
        delta = backwards(delta, params, 'layer3', relu_deriv)
        delta = backwards(delta, params, 'layer2', relu_deriv)
        backwards(delta, params, 'layer1', relu_deriv)

        # Apply gradient
        for layer in ['output','layer1','layer2','layer3']:
            params['m_W' + layer] = 0.9*params['m_W' + layer] - learning_rate * params['grad_W' + layer]
            params['W' + layer] += params['m_W' + layer] 
            params['m_b' + layer] = 0.9*params['m_b' + layer]  - learning_rate * params['grad_b' + layer]
            params['b' + layer]+= params['m_b' + layer] 
        
    train_loss.append(total_loss/train_x.shape[0])

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9
        
    
plt.figure()
plt.plot(range(max_iters), train_loss, color='g')
plt.legend(['training loss'])
plt.show()
# Q5.3.1
h1 = forward(valid_x, params, 'layer1', relu)
h2 = forward(h1, params, 'layer2', relu)
h3 = forward(h2, params, 'layer3', relu)
valid_out = forward(h3, params, 'output', sigmoid)
# visualize some results

# Q5.3.2
for i in range(5):
    plt.subplot(2,1,1)
    index = int(3600/5*i)
    plt.imshow(valid_x[index,:].reshape(32,32).T)
    plt.subplot(2,1,2)
    plt.imshow(valid_out[index,:].reshape(32,32).T)
    plt.show()
    plt.subplot(2, 1, 1)
    index = int(3600 / 5 * i+1)
    plt.imshow(valid_x[index, :].reshape(32, 32).T)
    plt.subplot(2, 1, 2)
    plt.imshow(valid_out[index, :].reshape(32, 32).T)
    plt.show()

from skimage.measure import compare_psnr as psnr
# evaluate PSNR
##########################
##### your code here #####
##########################
psnr_noisy = 0
for i in range(valid_out.shape[0]):
    psnr_noisy += psnr(valid_x[i], valid_out[i])

psnr_noisy /= valid_out.shape[0]
print(psnr_noisy)
