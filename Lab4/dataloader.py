
import os
from glob import glob
import torch
from torch import stack
from torch.utils.data import Dataset as torchData

from torchvision.datasets.folder import default_loader as imgloader
from torch import stack
def get_key(fp):
    filename = fp.split('/')[-1]
    filename = filename.split('.')[0].replace('frame', '')

    return int(filename)

class Dataset_Dance(torchData):
    """
        Args:
            root (str)      : The path of your Dataset
            transform       : Transformation to your dataset
            mode (str)      : train, val, test
            partial (float) : Percentage of your Dataset, may set to use part of the dataset
    """
    def __init__(self, root, transform, mode='train', video_len=7, partial=1.0):
        super().__init__()
        assert mode in ['train', 'val'], "There is no such mode !!!"
        if mode == 'train':
            png_file = glob(os.path.join(root, 'train/train_img/*.png'))
            self.img_folder  = sorted((os.path.normpath(file_path).replace("\\","/") for file_path in png_file), key=get_key)
            self.prefix = 'train'
        elif mode == 'val':
            png_file = glob(os.path.join(root, 'val/val_img/*.png'))
            self.img_folder  = sorted((os.path.normpath(file_path).replace("\\","/") for file_path in png_file), key=get_key)
            self.prefix = 'val'
        else:
            raise NotImplementedError
        
        self.transform = transform
        self.partial = partial
        self.video_len = video_len

    def __len__(self):
        return int(len(self.img_folder) * self.partial) // self.video_len

    def __getitem__(self, index):
        path = self.img_folder[index]
        
        imgs = []
        labels = []
        for i in range(self.video_len):
            label_list = self.img_folder[(index*self.video_len)+i].split('/')
            label_list[-2] = self.prefix + '_label'
            
            img_name    = self.img_folder[(index*self.video_len)+i]
            label_name = '/'.join(label_list)

            imgs.append(self.transform(imgloader(img_name)))
            labels.append(self.transform(imgloader(label_name)))
        return stack(imgs), stack(labels)
# if __name__ == '__main__':

#     from torch.utils.data import DataLoader
#     from torchvision import transforms
#     transform = transforms.Compose([
#         transforms.Resize((32,64)),
#         transforms.ToTensor()
#     ])
#     # 創建數據集實例
#     dataset = Dataset_Dance(root='C:/Users/Steven/Desktop/課程資料/交大/2023DL/Lab/Lab4/LAB4_Dataset/', transform=transform, mode='train')

#     # 創建 DataLoader
#     train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

#     # 在訓練迴圈中使用 DataLoader 加載數據
#     for batch_idx, (data, labels) in enumerate(train_loader):
#         print("Batch", batch_idx, "Data shape:", data.shape)
#         # 在這裡進行訓練操作，data 和 labels 是從 DataLoader 中讀取的批次數據