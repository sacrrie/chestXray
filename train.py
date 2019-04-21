#utils.py: various helper functions
#netModel.py: modified and improved nn model
from utils import *
from DenseModel import *
import os
import torchvision
import numpy as np
import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.nn import functional
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
    input_dir=IMG_DIR
    image_list=TRAIN
    transform=Compose([
        RandomResizedCrop(224),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize
        ])
    )
dataset_validation=XrayDataset(
    input_dir=IMG_DIR
    image_list=TRAIN
    transform=Compose([
        RandomResizedCrop(224),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize
        ])
    )
train_loader=DataLoader(dataset=dataset_train,batch_size=BATCH,shuffle=True,num_workers=16,pin_memory=True)
validation_loader=DataLoader(dataset=dataset_validation,batch_size=BATCH,shuffle=False,num_workers=16,pin_memory=True)

#setup loss function,optimizer and sheduler
loss=torch.nn.BCELoss(size_average=True)
optimizer=optim.Adam(model.parameters(),lr=0.0001,betas(0.9,0.999),eps=0.00000001,weight_decay=0.00001)
scheduler=ReduceLROnPlateau(optimizer,factor=0.1,patience=5,mode='min')
