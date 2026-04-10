import torch
import torch.nn as nn

from torchvision.models import resnet50, ResNet50_Weights


class CNN(nn.Module):
    def __init__(self, embed_dim, device):

        super().__init__()

        model = resnet50(weights = ResNet50_Weights.DEFAULT)

        modules = list(model.children())[:-1]
        
        size = 2048

        self.model_backbone = nn.Sequential(*modules)

        #Freeze the training 
        for i in self.model_backbone.parameters():
            i.requires_grad = False
        
        self.linear = nn.Linear(size, embed_dim)

        self.to(device)
    

    def forward(self,x):
        vector_unflattened = self.model_backbone(x)

        #Start at dim 1 to keep batch intact
        vector_flattened = torch.flatten(vector_unflattened, start_dim=1) 

        feature_vector = self.linear(vector_flattened)


        

        return feature_vector



class LSTM(nn.Module):

    def __init__(self, embed_dim, device, vocab_size): 

        super().__init__()

        hidden_size = 256


        self.encoder = nn.Embedding(vocab_size, embed_dim)
        self.model = nn.LSTM(
            input_size=embed_dim * 2,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first= True,
            device = device
        ).to(device)

        print(f"Vocab Size is :{vocab_size}")

        self.linear = nn.Linear (hidden_size,vocab_size)

        self.to(device)

    def forward(self, captions, img_features):
        if not isinstance(captions, torch.Tensor):
            captions = torch.stack(captions)


        embedding = self.encoder(captions)

        img_features = img_features.unsqueeze(1).repeat(1,embedding.size(1),1)
        x = torch.cat((img_features,embedding),dim = 1)
        
        output, (hn,cn) =self.model(x)

        logits =  self.linear(output)

        return logits
    









        