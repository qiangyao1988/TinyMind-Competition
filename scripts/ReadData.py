# -*- coding: utf-8 -*-
# @Time    : Thu Mar  7 13:43:52 2019
# @Author  : Yao Qiang
# @Email   : qiangyao1988wsu@gmail.com
# @File    : ReadData.py
# @Software: Spyder
# @Pythpon Version: python3.6


import os
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm


def loadOneWord(order):
    '''
    Create one word dataset
    
    Param:
        order: the order number in trainset folder
        
    Return:
        datas: data of this order word in type of numpy.ndarray
        labels: label of this order word in type of numpy.ndarray
    '''
    # get file path
    path = trainpath + '/' + words[order]
    files = os.listdir(path)
    
    datas = []
    
    # traverse the file folder to get all the image files
    for file in files:
        file = path + '/' + file
        img = np.asarray(Image.open(file))
        img = cv2.resize(img, img_size)
        datas.append(img)
    
    # save datas and labels as type of numpy.ndarray
    datas = np.array(datas)
    labels = np.zeros([len(datas), len(words)], dtype=np.uint8)
    labels[:, order] = 1
    return datas, labels


def transData():
    '''
    Transfer and save the datas

    '''
    num = len(words)
    datas = np.array([], dtype=np.uint8)
    datas.shape = -1, 128, 128
    labels = np.array([], dtype=np.uint8)
    labels.shape = -1, 100
    for k in tqdm(range(num)):
        data, label = loadOneWord(k)
        datas = np.append(datas, data, axis=0)
        labels = np.append(labels, label, axis=0)

    np.save('../dataset/data.npy', datas)
    np.save('../dataset/label.npy', labels)
    
    

def loadtestdata():
    '''
    Create test dataset
    
    '''
    files = os.listdir(testpath)
    datas = []
    for file in tqdm(files):
        file = testpath + '/' + file
        img = np.asarray(Image.open(file))
        img = cv2.resize(img, img_size)
        datas.append(img)
    datas = np.array(datas)
    return datas


if __name__ == '__main__':
    
    trainpath =  "../dataset/train"
    testpath = "../dataset/test1"
    
    words = os.listdir(trainpath)

    category_number = len(words)

    # img_size = (256, 256)
    img_size = (128, 128)
    
    transData()