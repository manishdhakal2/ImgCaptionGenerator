import torch
import torch.nn as nn
from encoder import Encoder
from models import CNN, LSTM
from loader import load_dataset

from train import train_models



if __name__ == "__main__":
    

    #Define hyperparams
    lr = 0.01
    epochs = 15

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


   

    tokenizer = Encoder()

    print("Loading dataloaders......")
    train_loader, test_loader = load_dataset(r"D:/Python/ImgCaptionGenerator/data/images",
                                              r"D:/Python/ImgCaptionGenerator/data/captions.txt",
                                                tokenizer)
    
    print("Dataloaders loaded successfully..../n")

    
    vocab_size = tokenizer.max_length
    embed_dim = 256

    print("Initializing CNN and LSTM....")

    cnn_model = CNN(embed_dim, device)
    lstm_model = LSTM(embed_dim, device, vocab_size=tokenizer.max_length)

    print("CNN and LSTM Initialized successfully")


    print("Starting Training.....")

    train_models(cnn_model, lstm_model, train_loader, epochs = epochs ,learning_rate=lr, pad_index= tokenizer.pad_index)





