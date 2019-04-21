import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score

class XrayDataset(Dataset):
    def __init__(self, input_dir,image_list,transform=transforms.Resize(512)):
        """
        Arguments:
        input_dir: the directory contains image
        image_list: the file contains image names and their labels
        transform: transform applying to a sample,defaults to resize the images to 512x512 resolution
        """
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
        return(len(self.image_names))
        
    def __getitem__(self,index):
        name=self.image_names[index]
        image=Image.open(name).convert('RGB')
        image=self.transform(image)
        label=self.labels[index]
        return(image,torch.FloatTensor(label))

def get_AUCs(ground_truth,predicitons):
    results=[]
    truth=ground_truth.cpu().numpy()
    pred=predicitons.cpu().numpy()
    count=truth.shape[1]
    count_check=pred.shape[1]
    assert count==count_check
    for i in range(count):
        results.append(roc_auc_score(truth[:,i],pred[:,i]))
        
    return(results)