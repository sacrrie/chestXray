from torch import nn
from torchvision import densenet121

class DenseNetImproved(nn.Module):
    """
    At the end of the classifier layer, an additional <sigmoid function> was added to improve the model performance
    """
    def __init__(self,out_count):
        super(DenseNet121,self).__init__()
        self.densenet121=densenet121(pretrained=True)
        feature_count=self.densenet121.classifier.in_features
        self.densenet121.classifier=nn.Sequential(
            nn.Linear(feature_count,out_count),
            nn.sigmoid())
    
    def forward(self,x):
        x=self.densenet121(x)
        return x
