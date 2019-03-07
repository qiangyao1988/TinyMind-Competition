# -*- coding: utf-8 -*-
# @Time    : Thu Mar  7 14:22:56 2019
# @Author  : Yao Qiang
# @Email   : qiangyao1988wsu@gmail.com
# @File    : Test.py
# @Software: Spyder
# @Pythpon Version: python3.6


import pandas as pd
import os
import torch
from torch.autograd import Variable
import numpy as np
import CreateModel
from tqdm import tqdm
import cv2
from PIL import Image

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
    
    img_size = (128, 128)

    filename = os.listdir(testpath)
    words = os.listdir(trainpath)   # 按时间排序 从早到晚
    words = np.array(words)
    testnumber = len(filename)
    category_number = len(words)

    net = CreateModel.net()

    if torch.cuda.is_available():
        net.cuda()
    
    # eval model   
    net.eval()
    
    
    checkpoint = CreateModel.load_checkpoint()
    net.load_state_dict(checkpoint['state_dict'])

    testdatas = loadtestdata()
    testdatas.astype(np.float)
    
    n = 0
    N = 10000
    batch_size = 8
    pre = np.array([])
    batch_site = []
    
    while n < N:
        n += batch_size
        if n < N:
            n1 = n - batch_size
            n2 = n
        else:
            n1 = n2
            n2 = N

        batch_site.append([n1, n2])

    pred_choice = []
    
    for site in tqdm(batch_site):
        test_batch = testdatas[site[0]:site[1]]
        test_batch = torch.from_numpy(test_batch)
        datas = Variable(test_batch).cuda().float()
        datas = datas.view(-1, 1, 128, 128)
        outputs = net(datas)
        outputs = outputs.cpu()
        outputs = outputs.data.numpy()
        for out in outputs:
            K = 5
            index = np.argpartition(out, -K)[-K:]
            pred_choice.append(index)
            
    pre = np.array(pred_choice)
    predicts = []
    
    for k in range(testnumber):
        index = pre[k]
        predict5 = words[index]
        predict5 = "".join(predict5)
        predicts.append(predict5)

    dataframe = pd.DataFrame({'filename': filename, 'label': predicts})
    dataframe.to_csv("test.csv", index=False, encoding='utf-8')

    read = pd.read_csv('test.csv')
    print(read)