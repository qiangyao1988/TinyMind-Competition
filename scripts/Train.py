# -*- coding: utf-8 -*-
# @Time    : Thu Mar  7 13:49:02 2019
# @Author  : Yao Qiang
# @Email   : qiangyao1988wsu@gmail.com
# @File    : Train.py
# @Software: Spyder
# @Pythpon Version: python3.6


import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import CreateModel 
import TrainSet 

def train(epoch):
    '''
    train model
    '''
    
    # The net is in training model, so we can use drop out
    net.train() 
    correct = 0
    sum = 0
    T = 0
    
    
    running_loss = 0.0

    for batch_index, (datas, labels) in enumerate(trainloader, 0):
        labels = labels.max(1)[1]
        datas = Variable(datas).float()
        datas = datas.view(-1, 1, 256, 256)
        labels = Variable(labels).long()
        
        if torch.cuda.is_available():
            datas = datas.cuda()
            labels = labels.cuda()
        
        # forward
        optimizer.zero_grad()
        outputs = net(datas)
        loss = criterion(outputs, labels)
        
        # back
        loss.backward()
        optimizer.step()
        
                    
        # print statistics
        running_loss += loss.item()
        
        T += 1
        pred_choice = outputs.data.max(1)[1]
        correct += pred_choice.eq(labels.data).cpu().sum()
        sum += len(labels)
        # accuracy = correct / sum
        
        if batch_index % 100 == 99:
            print('[%d,%4d] loss: %.3f' %
                        (epoch + 1, batch_index + 1, running_loss))
            running_loss = 0.0
            print('Accuracy of the network: %d %%' % (100 * correct / sum))
            
        '''    
        print('batch_index: [%d/%d]' % (batch_index, len(trainloader)),
              'Train epoch: [%d]' % (epoch),
              'correct/sum:[%d/%d]' % (correct, sum),
              'accuracy:[%d]' % (accuracy))
        '''




def eval(epoch):
    '''
    eval mode
    '''
    # The net is in eval model, so we can not use drop out, and stop backpropagation
    net.eval()  
    correct = 0
    sum = 0
    
    for batch_index, (datas, labels) in enumerate(evalloader, 0):
        labels = labels.max(1)[1]
        datas = Variable(datas).cuda().float()
        datas = datas.view(-1, 1, 256, 256)
        labels = Variable(labels).cuda().long()
        # optimizer.zero_grad()
        outputs = net(datas)
        # loss = criterion(outputs, labels)
        # loss.backward()
        # optimizer.step()

        pred_choice = outputs.data.max(1)[1]
        correct += pred_choice.eq(labels.data).cpu().sum()
        sum += len(labels)
        
        # accuracy = correct / sum
        ''' 
        print('batch_index: [%d/%d]' % (batch_index, len(evalloader)),
              'Eval epoch: [%d]' % (epoch),
              'correct/sum:%d/%d' % (correct, sum),
              'accuracy:[%d]' % (accuracy))
        ''' 

if __name__ == '__main__':
    
    n_epoch, batch_size = 1, 8
    
    # create trainloader and evalloader
    trainset = TrainSet.TrainSet(eval=False)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    evalset = TrainSet.TrainSet(eval=True)
    evalloader = torch.utils.data.DataLoader(evalset, batch_size=batch_size, shuffle=True)
    
    net = CreateModel.net()
    
    if torch.cuda.is_available():
        net.cuda()
    
    # loss function   
    criterion = nn.CrossEntropyLoss()

    #optimizer
    optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)

    
    
    # Whether to load model parameters
    load = False

    if load:
        checkpoint = CreateModel.load_checkpoint()
        net.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0


    for epoch in range(start_epoch, n_epoch):
        train(epoch)

        # save checkpoint
        checkpoint = {'epoch': epoch, 'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()}
        CreateModel.save_checkpoint(checkpoint)

        eval(epoch)