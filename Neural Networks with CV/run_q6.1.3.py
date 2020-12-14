#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 19:21:12 2020

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
import torchvision
import torchvision.transforms as transforms

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

class Model(nn.Module):
    """CNN."""

    def __init__(self):
        """CNN Builder."""
        super(Model, self).__init__()

        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )


    def forward(self, x):
        """Perform forward."""
        
        # conv layers
        x = self.conv_layer(x)
        
        # flatten
        x = x.view(x.size(0), -1)
        
        # fc layer
        x = self.fc_layer(x)

        return x
    
model=Model()
device=torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
model = nn.DataParallel(model,device_ids=[4,5,6,7]).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs=50

def train(num_epochs,train_loader):
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
            
            print('E: %d, B: %d / %d, Train Loss: %.3f | avg_loss: %.3f, Time Taken : %.3f' % (epoch+1, batch_idx+1, 
                                                                           len(train_loader),
                  loss.item(),train_loss/(batch_idx+1),stop-start),end='\n ')
            
            if(epoch==num_epochs-1):
                torch.save(model.state_dict(), "q_6.1.2.pth")
            
            del loss
            del y
            del x
        train_loss_final.append(train_loss/len(train_loader))
        train_acc_final.append(total_accuracy/total)
        
        print("Train accuracy :",total_accuracy/total)
        
    return train_loss_final,train_acc_final

train_loss_list,train_acc_list=train(num_epochs,trainloader)

plt.figure('Loss')
plt.plot(np.arange(num_epochs),train_loss_list,'r')
plt.legend(['training loss'])
plt.show()

plt.figure('Accuracy')
plt.plot(np.arange(num_epochs),train_acc_list,'b')
plt.legend(['training accuracy'])
plt.show()

