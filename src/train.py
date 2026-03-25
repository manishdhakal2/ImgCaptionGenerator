from models import CNN, LSTM
import torch
import torch.nn as nn

from loader import load_dataset


def train_models(CNN, LSTM, train_loader, epochs, learning_rate, pad_index):


    optimizer = torch.optim.SGD(params = LSTM.parameters(), lr= learning_rate, momentum=0.08, weight_decay= 1e-4)

    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_index)

    for epoch in range(epochs):

        running_loss = 0.0

        LSTM.train()
        
        for batch in train_loader:

            images, input_captions, target_captions  = batch[0], batch[1], batch[2]

            img_features = CNN(images)

            y_pred = LSTM(images,input_captions)

            loss = loss_fn(y_pred, target_captions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss
        
        print(f"Epoch {epoch} Running Loss : {running_loss}")
            

