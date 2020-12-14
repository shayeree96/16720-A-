import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import string

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

max_iters = 50
# pick a batch size, learning rate
batch_size =32
learning_rate = 3e-4
hidden_size = 128

##########################
##### your code here #####
##########################

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here

initialize_weights(train_x.shape[1],hidden_size,params,'layer1')

initial_W = params['Wlayer1']
fig0 = plt.figure()
grid = ImageGrid(fig0, 111, nrows_ncols=(8,8,),axes_pad=0.0)
for i in range(64):
    grid[i].imshow(initial_W[:,i].reshape((32,32)))    
initialize_weights(hidden_size,train_y.shape[1],params,'output')


avg_acc=0
train_acc_list,valid_acc_list=[],[]
train_loss_list,valid_loss_list=[],[]
# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for xb,yb in batches:
        # training loop can be exactly the same as q2!
        
        h1 = forward(xb, params, 'layer1', sigmoid)
        probs = forward(h1, params, 'output', softmax)

        # Loss and accuracy
        loss, acc = compute_loss_and_acc(yb, probs)
        total_loss += loss
        total_acc += acc

        # Backward pass
        delta1 = probs
        delta1[np.arange(probs.shape[0]), np.argmax(yb, axis=1)] -= 1
        delta = backwards(delta1, params, 'output', linear_deriv)
        backwards(delta, params, 'layer1', sigmoid_deriv)

        # Apply gradient
        for layer in ['output', 'layer1']:
            params['W' + layer] -= learning_rate * params['grad_W' + layer]
            params['b' + layer] -= learning_rate * params['grad_b' + layer]

    # Total accuracy
    avg_acc = total_acc / batch_num
    total_loss = total_loss / train_x.shape[0]

    print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,avg_acc))

# run on validation set and report accuracy! should be above 75%

    # Validation forward pass
    valid_h1 = forward(valid_x, params, 'layer1', sigmoid)
    valid_probs = forward(valid_h1, params, 'output', softmax)
    
    # Validation loss and accuracy
    valid_loss, valid_acc = compute_loss_and_acc(valid_y, valid_probs)
    valid_loss /= valid_x.shape[0]
    
    train_loss_list.append(total_loss)
    valid_loss_list.append(valid_loss)
    train_acc_list.append(avg_acc)
    valid_acc_list.append(valid_acc)
    
    print('Validation accuracy: ',valid_acc)
    
    if False: # view the data
        for crop in xb:
            import matplotlib.pyplot as plt
            plt.imshow(crop.reshape(32,32).T)
            plt.show()
           
plt.figure(0)
plt.plot(np.arange(max_iters),train_loss_list,'r')
plt.plot(np.arange(max_iters),valid_loss_list,'b')
plt.legend(['training loss','valid loss'])
plt.figure(1)
plt.plot(np.arange(max_iters),train_acc_list,'r')
plt.plot(np.arange(max_iters),valid_acc_list,'b')
plt.legend(['training accuracy','valid accuracy'])
plt.show()

#We finally ru on the test set

test_h1 = forward(test_x, params, 'layer1', sigmoid)
test_probs = forward(test_h1, params, 'output', softmax)

# Validation loss and accuracy
test_loss, test_acc = compute_loss_and_acc(test_y, test_probs)
test_loss /= test_y.shape[0]

print('Test Accuracy:', test_acc)

import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.3

learned_W = params['Wlayer1']
fig3 = plt.figure()
grid = ImageGrid(fig3, 111, nrows_ncols=(8,8,),axes_pad=0.0)
for i in range(64):
    grid[i].imshow(learned_W[:,i].reshape((32,32)))

# Q3.4
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))

# compute comfusion matrix here

confusion_matrix = np.zeros((test_y.shape[1], test_y.shape[1]))

for i in range(test_probs.shape[0]):
    if np.argmax(test_y[i,:]).astype('int')==np.argmax(test_probs[i,:]).astype('int'):
        true=np.argmax(test_y[i,:]).astype('int')
        pred=np.argmax(test_probs[i,:]).astype('int')
        confusion_matrix[true, pred] += 1

plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()
