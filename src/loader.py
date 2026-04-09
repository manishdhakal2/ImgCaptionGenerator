import torch
import numpy as np
import cv2
import os

from models import CNN
from encoder import Encoder

from sklearn.model_selection import train_test_split  

from torch.utils.data import Dataset, DataLoader



def load_dataset(cnn_model, img_path, caption_path, tokenizer, device):


    """"
    Returns  (train Dataloader, test Dataloader, Instance of Encoder class)
    
    """


    

    

    with open(caption_path, "r+") as caption_file:
        captions = caption_file.read()

    caption_list=captions.split("\n")[1:]

    caption_dict = {  key:value.strip()  
                    for element in caption_list
                    if "," in element
                    for key,value in [element.split(",",1)]
                    
                    }
    

    

    
    image_list = []

    for key in caption_dict.keys():

        image_file = os.path.join(img_path, key)

        img = cv2.imread(image_file)

        if img is None:
            print(f"Couldn't Load {key}")
        

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        

        img = cv2.resize(img,(224,224))

        

            

        image_list.append(img)

    

    img_tensor = torch.tensor(np.array(image_list), dtype = torch.float32,device=device)/255.0
    img_tensor = img_tensor.permute(0,3,1,2)


    print("Forward pass of CNN in progress ...")

    batch_size = 32

    img_array = np.array(image_list)
    features_all = []

    cnn_model.eval()
    with torch.no_grad():
        for i in range(0, len(img_array), batch_size):
            batch = img_array[i:i+batch_size]
            batch = torch.tensor(batch, dtype=torch.float32).permute(0,3,1,2).to(device) / 255.0
            features = cnn_model(batch).cpu()  
            features_all.append(features)

    feature_vector = torch.cat(features_all, dim=0)



    print("Images successfully converted to feature vectors !")

    captions = list(caption_dict.values())

    tokenizer.build_vocab(captions)

    print("Vocabs built !")

    captions = tokenizer.encode_text(captions)
    


    captions = [torch.tensor(c,dtype=torch.long,device=device) for c in captions]

    captions = torch.stack(captions)

    train_img, test_img, train_caps, test_caps = train_test_split(feature_vector, captions, test_size = 0.33, random_state= 42)


    train_dataset = ImageCaptionDataset(img_tensor= train_img, labels = train_caps)
    test_dataset = ImageCaptionDataset(img_tensor= test_img, labels = test_caps)

    print("Dataset created !")


    train_loader = DataLoader(train_dataset, batch_size = 8, shuffle= True, num_workers= 0)
    test_loader = DataLoader(test_dataset, batch_size = 8, shuffle= False, num_workers= 0)
    print("DataLoaders created !")



    return train_loader, test_loader


    



class ImageCaptionDataset(Dataset):
    def __init__(self, img_tensor, labels):
        self.img = img_tensor

        
    
        self.caption = labels
    
    def __len__(self):

        return self.img.shape[0]
    
    def __getitem__(self,index):

        caption = self.caption[index]


        #Input caption
        input_cap = caption[:-1]
        target_cap = caption[1:]

        return self.img[index],input_cap, target_cap

   



            

