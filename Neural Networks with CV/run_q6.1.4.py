#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 21:29:23 2020

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
from os.path import join
from PIL import Image

num_epochs = 50
num_classes = 8
batch_size = 32
learning_rate = 0.01

data_dir = "./data/data/"
train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
train_labels = open(join(data_dir, 'train_labels.txt')).read().splitlines()
train_y = np.array([int(x) for x in train_labels])

test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
test_labels = open(join(data_dir, 'test_labels.txt')).read().splitlines()
test_y = np.array([int(x) for x in test_labels])

train_image = []
from skimage import transform
for i in range(len(train_files)):
    filename=data_dir+train_files[i]
    im=Image.open(filename)
    im = np.asarray(im)
    image = transform.resize(im, (224,224))
    train_image.append(image)

test_image = []
for i in range(len(test_files)):
    filename=data_dir+train_files[i]
    im=Image.open(filename)
    im = np.asarray(im)
    image = transform.resize(im, (224, 224))
    test_image.append(image)
    
train_x = np.array(train_image).transpose(0,3,1,2)
test_x = np.array(test_image).transpose(0,3,1,2)   

train_x = torch.Tensor(train_x)
train_y = torch.Tensor(train_y).type(torch.int64)
train_loader = torch.utils.data.TensorDataset(train_x,train_y)
train_loader = torch.utils.data.DataLoader(train_loader,batch_size=batch_size,shuffle=True)

test_x = torch.Tensor(test_x)
test_y = torch.Tensor(test_y).type(torch.int64)
test_loader = torch.utils.data.TensorDataset(test_x,test_y)
test_loader = torch.utils.data.DataLoader(test_loader,batch_size=batch_size,shuffle=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=8):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])

model=ResNet34()
device=torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
model = nn.DataParallel(model,device_ids=[4,5,6,7]).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
num_epochs=50

def train(num_epochs,train_loader,test_loader):
    train_loss_final=[]
    test_loss_final=[]
    train_acc_final=[]
    test_acc_final=[]
    
    
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
            '''
            print('E: %d, B: %d / %d, Train Loss: %.3f | avg_loss: %.3f, Time Taken : %.3f' % (epoch+1, batch_idx+1, 
                                                                           len(train_loader),
                  loss.item(),train_loss/(batch_idx+1),stop-start),end='\n ')
            '''
            if(epoch==num_epochs-1):
                torch.save(model.state_dict(), "q_6.1.4.pth")
            
            del loss
            del y
            del x
        train_loss_final.append(train_loss/len(train_loader))
        train_acc_final.append(total_accuracy/total)
        

        model.eval() 
        print("In evaluation -->")
        total_accuracy=0
        total=0

        for batch_idx,(x,y) in enumerate(test_loader):
            accuracy=0
            x=x.to(device)
            y=y.to(device)
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
            
            print('E: %d, B: %d / %d, Valid Loss: %.3f | avg_loss: %.3f, Time Taken : %.3f, Validation Accuracy : %.4f' % (epoch+1, batch_idx+1, 
                                                                           len(test_loader),
                  loss.item(),valid_loss/(batch_idx+1),stop-start,(accuracy*100)/len(y)),end='\n ')
            
            del x
            del y
        
        print("Test Accuracy is {} at Epoch {} ".format((total_accuracy*100)/total ,epoch+1))
        test_loss_final.append(valid_loss/len(test_loader))
        test_acc_final.append(total_accuracy/total)
        model.train()    
        
    return train_loss_final,test_loss_final,train_acc_final,test_acc_final

train_loss_list,test_loss_list,train_acc_list,test_acc_list=train(num_epochs,train_loader,test_loader)

plt.figure('Loss')
plt.plot(np.arange(num_epochs),train_loss_list,'r')
plt.legend(['training loss'])
plt.show()

plt.figure('Accuracy')
plt.plot(np.arange(num_epochs),train_acc_list,'b')
plt.legend(['training accuracy'])
plt.show()


