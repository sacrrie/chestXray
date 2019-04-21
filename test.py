#utils.py: various helper functions
#netModel.py: modified and improved nn model
from utils import *
from DenseModel import *
import os
import re
import torchvision
import numpy as np
import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.transforms import *

CP_PATH='model.pth.tar'
IMG_DIR='./images'
TEST='./labels/test_list.txt'
BATCH=128
NUM_DISEASES=14
DISEASES=[
    'Athelectasis','Cardiomegaly','Effusion','Infiltration',
    'Mass','Nodule','Pheumonia','Pheumothorax',
    'Consolidation','Edema','Emphysema','Fibrosis',
    'Pleural_Thickening','Hernia'
    ]


cudnn.benchmark=True
predictor=DenseNetImproved(NUM_DISEASES).cuda()
predictor=torch.nn.DataParallel(predictor).cuda()
#check saved checkpoint
if os.path.isfile(CP_PATH):
    print("checkpoint found, loading...")
    cp=torch.load(CP_PATH)
    #In case of loading to higher versions, update dictionary index
    state_dict=cp['state_dict']
    pattern=re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$'
        )
    for key in list(state_dict.keys()):
        res=pattern.match(key)
        if res:
            update=res.group(1)+res.group(2)
            state_dict[update]=state_dict[key]
            del state_dict[key]
    #Updating finished
    print("chekpoint loaded")
else:
    print("cannot find any checkpoint")
    
normalize=Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
processed=XrayDataset(
    input_dir=IMG_DIR,
    image_list=TEST,
    transform=Compose([
        Resize(256),
        TenCrop(224),
        Lambda(
            Lambda crops: torch.stack([
                ToTensor()(crop) for crop in crops
                ])),
        Lambda(
            Lambda crops: torch.stack([
                normalize(crop) for crop in crops
                ]))
        ])
    )
    
testset=DataLoader(dataset=processed,batch_size=BATCH,shuffle=False,num_workers=16,pin_memory=True)

truth=torch.FloatTensor().cuda()
predicted=torch.FloatTensor().cuda()
predictor.eval()
with torch.no_grad():
    for i,(inp,target) in enumerate(testset):
        target=target.cuda()
        truth=torch.cat((truth,target),0)
        bs,n_crops,c,h,w=inp.size()
        inputs=torch.autograd.Variable(inp.view(-1,c,h,w).cuda(),volatile=True)
        outputs=predictor(inputs).view(bs,n_crops,-1).mean(1).data
        predicted=torch.cat((predicted,outputs),0)
#TODO:添加一些图片输出和其他相关数据，F1分数，混淆矩阵啊之类的
auroc=get_AUCs(truth,predicted)
roc_avg=np.array(auroc).mean()
print('The average ROC is {}',roc_avg)
for i in range(NUM_DISEASES):
    print('The ROC of {} is {}'.format(DISEASES[i],auroc[i]))
