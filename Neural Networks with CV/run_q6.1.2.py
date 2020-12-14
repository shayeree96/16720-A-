#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 19:18:13 2020

@author: shayereesarkar
"""
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import random
import copy
import time
import torch.nn.functional as F
import pandas as pd
from pandas import DataFrame
import scipy.io
import matplotlib.pyplot as plt

train_data = scipy.io.loadmat('./data/nist36_train.mat')
valid_data = scipy.io.loadmat('./data/nist36_valid.mat')
test_data = scipy.io.loadmat('./data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

train_x = torch.from_numpy(train_x).float()
train_y = torch.from_numpy(train_y).long()
valid_x = torch.from_numpy(valid_x).float()
valid_y = torch.from_numpy(valid_y).long()

batch_size =50
learning_rate = 0.001

train_loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y),shuffle=True,batch_size=batch_size,num_workers=36,pin_memory=True)
valid_loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(valid_x, valid_y),shuffle=False,batch_size=batch_size,num_workers=36,pin_memory=True)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,10,5),
            nn.MaxPool2d(2,2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10,20,5),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(2,2),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(500,200),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Linear(200,36)
    
        

    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f2_flat = f2.view(f2.size(0), -1)
        f3 = self.fc1(f2_flat)
        output = self.fc2(f3)
        return output 

model=Model()
device=torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
model = nn.DataParallel(model,device_ids=[4]).to(device)

num_epochs=15
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate )

def train(num_epochs,train_loader,valid_loader):
    train_loss_final=[]
    valid_loss_final=[]
    train_acc_final=[]
    valid_acc_final=[]
    
    
    for epoch in range(num_epochs):
        model.train()
        #print(' epoch :',epoch+1)
        train_loss=0
        valid_loss=0
        total_accuracy=0
        total=0
        for batch_idx,(x,y) in enumerate(train_loader):
            start=time.time()
            accuracy=0
            x=x.to(device)
            y=y.to(device)
            y = torch.max(y, 1)[1]
            x = x.reshape(-1,1,32,32)

            output=model(x)
            loss=criterion(output,y)
            optimizer.zero_grad()
            loss.backward()        
            optimizer.step()
            train_loss+=loss.item()
            
            predictions=F.softmax(output,dim=1)
            _,top_prediction=torch.max(predictions,1)#To get the 
            top_pred_labels=top_prediction.view(-1)
            accuracy+=torch.sum(torch.eq(top_pred_labels,y)).item()
            total_accuracy+=accuracy
            total+=len(y)
            stop=time.time()
            '''
            print('E: %d, B: %d / %d, Train Loss: %.3f | avg_loss: %.3f, Time Taken : %.3f' % (epoch+1, batch_idx+1, 
                                                                           len(train_loader),
                  loss.item(),train_loss/(batch_idx+1),stop-start),end='\n ')
            '''
            if(epoch==num_epochs-1):
                torch.save(model.state_dict(), "q_6.1.2.pth")
            
            del loss
            del y
            del x
        train_loss_final.append(train_loss/len(train_loader))
        train_acc_final.append(total_accuracy/total)
        

        model.eval() 
        print("In evaluation -->")
        total_accuracy=0
        total=0

        for batch_idx,(x,y) in enumerate(valid_loader):
            accuracy=0
            x=x.to(device)
            y=y.to(device)
            y = torch.max(y, 1)[1]
            x = x.reshape(-1,1,32,32)
            
            output=model(x)
            loss=criterion(output,y)
            valid_loss+=loss.item()
            #We have to calculate accuracy
            predictions=F.softmax(output,dim=1)
            _,top_prediction=torch.max(predictions,1)#To get the 

            top_pred_labels=top_prediction.view(-1)
            accuracy+=torch.sum(torch.eq(top_pred_labels,y)).item()
            total+=len(y)
            total_accuracy+=accuracy
            '''
            print('E: %d, B: %d / %d, Valid Loss: %.3f | avg_loss: %.3f, Time Taken : %.3f, Validation Accuracy : %.4f' % (epoch+1, batch_idx+1, 
                                                                           len(valid_loader),
                  loss.item(),valid_loss/(batch_idx+1),stop-start,(accuracy*100)/len(y)),end='\n ')
            '''
            del x
            del y
        
        print(" Validation Accuracy is {} at Epoch {} ".format((total_accuracy*100)/total ,epoch+1))
        valid_loss_final.append(valid_loss/len(valid_loader))
        valid_acc_final.append(total_accuracy/total)
        model.train()    
        
    return train_loss_final,valid_loss_final,train_acc_final,valid_acc_final

train_loss_list,valid_loss_list,train_acc_list,valid_acc_list=train(num_epochs,train_loader,valid_loader)

plt.figure('Loss')
plt.plot(np.arange(num_epochs),train_loss_list,'r')
plt.plot(np.arange(num_epochs),valid_loss_list,'b')
plt.legend(['training loss','valid loss'])
plt.show()

plt.figure('Accuracy')
plt.plot(np.arange(num_epochs),train_acc_list,'r')
plt.plot(np.arange(num_epochs),valid_acc_list,'b')
plt.legend(['training accuracy','valid accuracy'])
plt.show()

