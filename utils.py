import os
import torch
import matplotlib.pyplot as plt
import matplotlib.cm
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score,f1_score,confusion_matrix

class XrayDataset(Dataset):
    def __init__(self, input_dir,image_list,transform=transforms.Resize(256)):

        file_names=[]
        labels=[]
        flist=open(image_list,"r")
        for record in flist:
            info=record.split()
            file_name=os.path.join(input_dir,info[0])
            label=list(map(int,info[1:]))
            file_names.append(file_name)
            labels.append(label)
        self.file_names=file_names
        self.labels=labels
        self.transform=transform
        
    def __len__(self):
        return(len(self.file_names))
        
    def __getitem__(self,index):
        name=self.file_names[index]
        image=Image.open(name).convert('RGB')
        image=self.transform(image)
        label=self.labels[index]
        return(image,torch.FloatTensor(label))

def get_metrics(ground_truth,predicitons):
    roc=[]
    truth=ground_truth.cpu().numpy()
    pred=predicitons.cpu().numpy()
    count=truth.shape[1]
    count_check=pred.shape[1]
    assert count==count_check
    for i in range(count):
        roc.append(roc_auc_score(truth[:,i],pred[:,i]))
        
    threshold=0.95
    pred=np.where(pred>threshold,1,0)
    truth=np.argmax(truth,axis=1)
    pred=np.argmax(pred,axis=1)
    f1=f1_score(truth,pred,average='weighted')

    return(roc,f1)

