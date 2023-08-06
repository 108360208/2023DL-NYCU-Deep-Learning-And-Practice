import pandas as pd
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms
import os
def getData(mode):
    if mode == 'train':
        df = pd.read_csv('train.csv')
        path = df['Path'].tolist()
        label = df['label'].tolist()
        return path, label
    
    elif mode == "valid":
        df = pd.read_csv('valid.csv')
        path = df['Path'].tolist()
        label = df['label'].tolist()
        return path, label
    
    else:
        df = pd.read_csv('resnet_18_test.csv')
        path = df['Path'].tolist()
        label = [0] * len(path)
        return path,label

class LeukemiaLoader(data.Dataset):
    def __init__(self,root,mode,transform=None):
        self.root = root
        self.img_name, self.labels = getData(mode)
        self.transform = transform
        self.mode = mode
        print("> Found %d images..."%(len(self.img_name)))
    def __len__(self):
        return len(self.img_name)
    def __getitem__(self,idx):
        img_name = os.path.join(self.root, os.path.normpath(self.img_name[idx]))

        image = Image.open(img_name).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)

        return image, label

resnet18_train__data_transform = transforms.Compose([
        #transforms.CenterCrop(300),
        transforms.Resize((50,50)),  # 保持图像原始大小
        transforms.RandomHorizontalFlip(),  # 隨機水平翻轉
        transforms.RandomRotation(15),  # 隨機旋轉圖像
        #transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.15, saturation=0.1, hue=0.1)],p= 0.5),  # 添加色彩增強，以50%的機率進行處理
        #transforms.RandomApply([transforms.GaussianBlur(kernel_size=5,sigma=(1.8,2))], p=0.5),  # 添加高斯模糊，以20%的機率進行處理
        #transforms.RandomApply([transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10)], p=0.3), 
        #transforms.ColorJitter(brightness=0.2, contrast=0.5, saturation=0.2, hue=0.1),  # 色彩增強
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 正規化
])
resnet18_valid__data_transform = data_transform = transforms.Compose([
        transforms.CenterCrop(300),
        transforms.Resize((30,30)),  # 保持图像原始大小
        #transforms.ColorJitter(brightness=0.2, contrast=0.5, saturation=0.2, hue=0.1),  # 色彩增強
        transforms.ToTensor(), 
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 正規化
])
resnet50_train__data_transform = transforms.Compose([
        transforms.CenterCrop(300),
        transforms.Resize((30,30)),  # 保持图像原始大小
        transforms.RandomHorizontalFlip(),  # 隨機水平翻轉
        transforms.RandomRotation(15),  # 隨機旋轉圖像
        #transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.15, saturation=0.1, hue=0.1)],p= 0.5),  # 添加色彩增強，以50%的機率進行處理
        #transforms.RandomApply([transforms.GaussianBlur(kernel_size=5,sigma=(1.8,2))], p=0.5),  # 添加高斯模糊，以20%的機率進行處理
        #transforms.RandomApply([transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10)], p=0.3), 
        #transforms.ColorJitter(brightness=0.2, contrast=0.5, saturation=0.2, hue=0.1),  # 色彩增強
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 正規化
])
resnet50_valid__data_transform = data_transform = transforms.Compose([
        #transforms.CenterCrop(300),
        #transforms.Resize((30,30)),  # 保持图像原始大小
        #transforms.ColorJitter(brightness=0.2, contrast=0.5, saturation=0.2, hue=0.1),  # 色彩增強
        transforms.ToTensor(), 
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 正規化
])
resnet152_train__data_transform = transforms.Compose([
        transforms.CenterCrop(300),
        transforms.Resize((80, 80)),  # 保持圖像原始大小
        transforms.RandomHorizontalFlip(),  # 隨機水平翻轉
        transforms.RandomRotation(15),  # 隨機旋轉圖像
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 添加色彩增強，以50%的機率進行處理
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3,sigma=(1.8,2.2))], p=0.1),  # 添加高斯模糊，以20%的機率進行處理
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 正規化
])
resnet152_valid__data_transform = transforms.Compose([
    transforms.CenterCrop(300),
    transforms.Resize((80, 80)),
    #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 正規化
])
def transform(net_type):
    if(net_type == "resnet18"):
        return {
            "train" : resnet18_train__data_transform,
            "valid" : resnet18_valid__data_transform
        }
    elif(net_type == "resnet50"):
        return {
            "train" : resnet50_train__data_transform,
            "valid" : resnet50_valid__data_transform
        }
    else:
        return {
            "train" : resnet152_train__data_transform,
            "valid" : resnet152_valid__data_transform
        }