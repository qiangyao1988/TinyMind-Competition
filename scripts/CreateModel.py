# -*- coding: utf-8 -*-
# @Time    : Thu Mar  7 13:47:40 2019
# @Author  : Yao Qiang
# @Email   : qiangyao1988wsu@gmail.com
# @File    : CreateModel.py
# @Software: Spyder
# @Pythpon Version: python3.6


import torch
import torch.utils.data as data
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import os


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        
        # conv1
        self.conv1 = nn.Sequential(
                    nn.Conv2d(1, 32, 7, stride=2, padding=3),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.MaxPool2d(2))
        
        # conv2
        self.conv2 = nn.Sequential(
                    nn.Conv2d(32, 32, 3, stride=1, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU())      
        
        # conv3
        self.conv3 = nn.Sequential(
                    nn.Conv2d(32, 64, 3, stride=1, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(2))
         
        # conv4
        self.conv4 = nn.Sequential(
                    nn.Conv2d(64, 128, 3, stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU())
        
        # conv5
        self.conv5 = nn.Sequential(
                    nn.Conv2d(128, 128, 3, stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d(2))             
        
        # fc1
        self.fc1 = nn.Sequential(
                    nn.Linear(128*8*8, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU())
        
        # fc2
        self.fc2 = nn.Sequential(
                    nn.Linear(1024, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU())
        
        # fc3
        self.fc3 = nn.Sequential(
                    nn.Linear(256, 100),
                    nn.BatchNorm1d(100),
                    nn.ReLU())
        
        # dropout
        self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(-1, 128 * 8 * 8)   # reshape in a vector
        x = self.drop(x)
        x = self.fc1(x)
        x = self.fc2(x)
        out = self.fc3(x)
        return out
    
    def _initialize_weights(self):
        
        # print(self.modules())
        for m in self.modules():
            # print(m)
            if isinstance(m, nn.Linear):
                # print(m.weight.data.type())
                nn.init.xavier_normal_(m.weight, gain=1)
                # print(m.weight)


def paramsshow(net):
    '''
    function to show parameters
    '''
    print(net)
    params = list(net.parameters())
    print("lenghth of parameters:",len(params))
    for name,parameters in net.named_parameters():
        print(name,':',parameters.size())


def save_checkpoint(state, save_adress='model_save'):
    '''
    save checkpoint
    '''
    name = 'model_parameters.pth.tar'

    folder = os.path.exists(save_adress)
    
    if not folder:
        os.mkdir(save_adress)
        print('--- create a new folder ---')
        
    fulladress = save_adress + '/' + name
    torch.save(state, fulladress)
    print('model saved:', fulladress)


def load_checkpoint(save_adress='model_save'):
    '''
    load checkpoint
    '''
    name = 'model_parameters.pth.tar'
    fulladress = save_adress + '/' + name
    return torch.load(fulladress)


if __name__ == '__main__':
    net = net()
    paramsshow(net)           

