from torch import nn
import torchvision

class DenseNetImproved(nn.Module):

    def __init__(self,out_count):
        super(DenseNetImproved,self).__init__()
        self.densenet121=torchvision.models.densenet121(pretrained=True)
        feature_count=self.densenet121.classifier.in_features
        self.densenet121.classifier=nn.Sequential(
            nn.Linear(feature_count,out_count),
            #nn.Tanh())
            nn.Sigmoid())
    
    def forward(self,x):
        x=self.densenet121(x)
        return x

