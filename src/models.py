import torch
import torch.nn as nn

from torchvision.models import resnet50, ResNet50_Weights


class CNN(nn.Module):
    def __init__(self,device):

        super().__init__()

        model = resnet50(weights = ResNet50_Weights.DEFAULT)

        modules = list(model.children())[:-1]
        
        size = None

        embed_dim = None

        self.model_backbone = nn.Sequential(*modules)

        #Freeze the training 
        for i in self.model_backbone.parameters():
            i.requires_grad = False
        
        self.linear = nn.Linear(size, embed_dim)
    

    def forward(self,x):
        vector_unflattened = self.model_backbone(x)

        #Start at dim 1 to keep batch intact
        vector_flattened = torch.flatten(vector_unflattened, start_dim=1) 

        feature_vector = self.linear(vector_flattened)

        return feature_vector



class LSTM(nn.Module):

    def __init__(self,device):

        super().__init__()

        hidden_size = 64
        self.model = nn.LSTM(
            input_size=None,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first= True,
            device = device
        )

        self.linear = nn.Linear (hidden_size,None)

    def forward(self, x):
        output, (hn,cn) = self.model(x)

        return self.linear(output)



class Encoder:
    def __init__(self):
        pass


        