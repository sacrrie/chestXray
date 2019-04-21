#utils.py: various helper functions
#netModel.py: modified and improved nn model
from utils import *
from DenseModel import *
import os
import time
import torchvision
import numpy as np
import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.nn import functional
from torch.autograd import Variable
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import *

#set up gloabl variable, some parameters are subject to further optimization
CP=False
CP_PATH=''
IMG_DIR='./images'
#TEST='./labels/test_list.txt'
TRAIN='./labels/train_list.txt'
VALIDATION='./labels/val_list.txt'
BATCH=32
NUM_DISEASES=14
EPOCH=100
DISEASES=[
    'Athelectasis','Cardiomegaly','Effusion','Infiltration',
    'Mass','Nodule','Pheumonia','Pheumothorax',
    'Consolidation','Edema','Emphysema','Fibrosis',
    'Pleural_Thickening','Hernia'
    ]

#initialize model
model=DenseNetImproved(NUM_DISEASES).cuda()
model=torch.nn.DataParallel(model).cuda()


#Get data loaders ready
normalize=Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
dataset_train=XrayDataset(
    input_dir=IMG_DIR,
    image_list=TRAIN,
    transform=Compose([
        RandomResizedCrop(224),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize
        ])
    )
dataset_validation=XrayDataset(
    input_dir=IMG_DIR,
    image_list=VALIDATION,
    transform=Compose([
        RandomResizedCrop(224),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize
        ])
    )
train_loader=DataLoader(dataset=dataset_train,batch_size=BATCH,shuffle=True,num_workers=16,pin_memory=True)
validation_loader=DataLoader(dataset=dataset_validation,batch_size=BATCH,shuffle=False,num_workers=16,pin_memory=True)
print("Training and validation dataset loaded")
#setup loss function,optimizer and schedulers
loss=torch.nn.BCELoss(size_average=True)
optimizer=optim.Adam(model.parameters(),lr=0.0001,betas=(0.9,0.999),eps=0.00000001,weight_decay=0.00001)
scheduler=ReduceLROnPlateau(optimizer,factor=0.1,patience=5,mode='min')

#start training
print("All set, start training")
loss_min=100000
for e in range(EPOCH):
    model.train()
    for i,(input_array,target_array) in enumerate(train_loader):
        targets=target_array.cuda(async = True)
        inputs=Variable(input_array)
        t=Variable(targets)
        output=model(inputs)
        l=loss(output,t)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        
    #one epoch finishd, test the training results
    model.eval()
    l_val=0
    l_norm=0
    l_tensor_mean=0
    for i,(input_array,target_array) in enumerate(validation_loader):
        targets=target_array.cuda(async = True)
        inputs=Variable(input_array)
        t=Variable(targets)
        output=model(inputs)
        l=loss(output,t)
        l_tensor_mean+=l
        l_val+=l.data[0]
        l_norm+=1
    loss_val=l_val/l_norm
    loss_tensor=l_tensor_mean/l_norm
    
    #print out per epoch training results
    scheduler.step(loss_tensor.data[0])
    if loss_val < loss_min:
        loss_min=loss_val
        torch.save({'epoch':e+1,'state_dict':model.state_dict(),'loss':loss_min,'optimizer':optimizer.state_dict()},'cp.pth.tar')
        print('Model improved, checkpoint saved ! Epoch {}: loss is {}'.format(e+1,loss_val))
    else:
        print('No improving, continue training ... Epoch {}: loss is {}'.format(e+1,loss_val))
        