import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import json
import os
from torchvision import transforms


with open("objects.json" , 'r') as f:
    labels_mapping = json.load(f)

def getData(data_path):
    with open(data_path , 'r') as f:
        data = json.load(f)
    return data



class iclevrDataSet(Dataset):
    def __init__(self,root,mode):
        self.root = root
        self.labels_mapping = labels_mapping
        self.mode = mode
        if(mode == "train"):
            self.data = getData("train.json")
            self.transforms = transforms.Compose([
                transforms.Resize([64, 64]),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
            ])
            self.image_names = list(self.data.keys())
        elif(mode == "test" or mode == "new_test"):
            self.data = getData(mode+".json")
            #print(len(self.data))
        else : 
            raise ValueError(f"No {mode} mode!")
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if(self.mode == "train"):
            img_name = self.image_names[index]
            img_labels = self.data[img_name] #image data labels 
            img_path = os.path.join(self.root,img_name)
            image = Image.open(img_path).convert('RGB')
            image = self.transforms(image)
            one_hot_label = np.zeros(len(self.labels_mapping), dtype=int)
            #print(image.shape)
            for label in img_labels:
                one_hot_label[self.labels_mapping[label]] = 1

        elif(self.mode == "test" or self.mode == "new_test"):
            image = torch.randn(3, 64, 64)
            labels = self.data[index]
            one_hot_label = np.zeros(len(self.labels_mapping), dtype=int)
            for label in labels:
                one_hot_label[self.labels_mapping[label]] = 1
       
        return image, torch.tensor(one_hot_label)

test = iclevrDataSet("iclevr","new_test")
# a,b= test.__getitem__(3)
# print(b)