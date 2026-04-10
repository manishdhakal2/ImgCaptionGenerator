from models import LSTM
import torch
from tqdm import tqdm
import torch.nn as nn

from loader import load_dataset


def train_models(cnn_model, lstm_model, train_loader, epochs, learning_rate, pad_index, device):

    pad_index = torch.tensor(pad_index)


    optimizer = torch.optim.SGD(params = lstm_model.parameters(), lr= learning_rate, momentum=0.9, weight_decay= 1e-4)

    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_index)

    for epoch in range(epochs):


        print(f"Starting epoch {epoch}")

        running_loss = 0.0

        lstm_model.train()

        batch_count = 0

        progress = tqdm(train_loader)
        for batch_count, batch in enumerate(progress):


            img_features, input_captions, target_captions  = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            input_captions = [torch.tensor(caption,dtype=torch.long,device=device) for caption in input_captions]

            

            y_pred = lstm_model(input_captions, img_features)
           

            #Permute in order to match dims and leave out the first index as it's a processed image feature
            loss = loss_fn(y_pred[:,1:,:].permute(0,2,1),target_captions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            progress.set_postfix({
        'loss': f'{loss.item():.4f}',
        'batch': batch_count
    })
        
        print(f"Epoch {epoch} Running Loss : {running_loss}")
            

