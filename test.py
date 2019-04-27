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

#set up parameters
CP_PATH='saved.pth.tar'#chekpoint file path
IMG_DIR='./images'
TEST='./labels/test_list.txt'
BATCH=32
NUM_DISEASES=14
DISEASES=['Athelectasis','Cardiomegaly','Consolidation','Edema','Effusion','Emphysema','Fibrosis','Hernia','Infiltration','Mass','Nodule','Pleural_Thickening','Pheumonia','Pheumothorax']
normalize=Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

#load the checkpoint file
cudnn.benchmark=True
predictor=DenseNetImproved(NUM_DISEASES).cuda()
predictor=torch.nn.DataParallel(predictor).cuda()
if not os.path.isfile(CP_PATH):
    print("chekpoint file not found!")
    exit()
cp=torch.load(CP_PATH)
state_dict=cp['state_dict']
predictor.load_state_dict(state_dict)
print("chekpoint file loaded")
#Notes for usages: If you try to import a saved file from earlier versions, 
#it is needed to update the naming pattern, use the code provided below from the official
#pytorch documents
#check saved checkpoint
#if os.path.isfile(CP_PATH):
#    print("checkpoint found, loading...")
#    cp=torch.load(CP_PATH)
#    #In case of loading to higher versions, update dictionary index
#    state_dict=cp['state_dict']
#    pattern=re.compile(
#        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$'
#        )
#    for key in list(state_dict.keys()):
#        res=pattern.match(key)
#        if res:
#            update=res.group(1)+res.group(2)
#            state_dict[update]=state_dict[key]
#            del state_dict[key]
#    #Updating finished
#    print("chekpoint loaded")
#else:
#    print("cannot find any checkpoint")
    
testsets=XrayDataset(
    input_dir=IMG_DIR,
    image_list=TEST,
    transform=Compose([
        Resize(256),
        TenCrop(224),
        Lambda(
            lambda crops: torch.stack([
                ToTensor()(crop) for crop in crops
                ])),
        Lambda(
            lambda crops: torch.stack([
                normalize(crop) for crop in crops
                ]))
        ])
    )
    
testloader=DataLoader(dataset=testsets,batch_size=BATCH,shuffle=False,num_workers=8,pin_memory=True)

truth=torch.FloatTensor().cuda()
predicted=torch.FloatTensor().cuda()

predictor.eval()
with torch.no_grad():
    for i,(inp,target) in enumerate(testloader):
        target=target.cuda()
        truth=torch.cat((truth,target),0)
        bs,n_crops,c,h,w=inp.size()
        inputs=torch.autograd.Variable(inp.view(-1,c,h,w).cuda())
        outputs=predictor(inputs).view(bs,n_crops,-1).mean(1).data
        predicted=torch.cat((predicted,outputs),0)

auroc,f1=get_metrics(truth,predicted)
roc_avg=np.array(auroc).mean()
print('The F1 score is {}'.format(f1))
print('The average ROC score is {}'.format(roc_avg))
for i in range(NUM_DISEASES):
    print('The ROC score of {} is {}'.format(DISEASES[i],auroc[i]))
