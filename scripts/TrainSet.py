# -*- coding: utf-8 -*-
# @Time    : Thu Mar  7 14:52:23 2019
# @Author  : Yao Qiang
# @Email   : qiangyao1988wsu@gmail.com
# @File    : TrainSet.py
# @Software: Spyder
# @Pythpon Version: python3.6


import numpy as np
import torch
import torch.utils.data as data


class TrainSet(data.Dataset):
    '''
    Create data loader
    '''
    def __init__(self, eval=False):
        
        # load data and label
        datas = np.load('../dataset/data.npy')
        labels = np.load('../dataset/label.npy')
        
        index = np.arange(0, len(datas), 1, dtype=np.int)
        
        # set random seed to make sure everytime we get the same subset
        np.random.seed(123)
        np.random.shuffle(index)
        
        # if eval is true, get 10% of data as cross validation dataset
        if eval:
            index = index[:int(len(datas) * 0.1)]
        else:
            index = index[int(len(datas) * 0.1):]
            
        self.data = datas[index]
        self.label = labels[index]
        np.random.seed()

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index]),torch.from_numpy(self.label[index])

    def __len__(self):
        return len(self.data)
    