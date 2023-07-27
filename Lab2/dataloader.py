import numpy as np
import matplotlib.pyplot as plt
def read_bci_data():
    S4b_train = np.load('S4b_train.npz')
    X11b_train = np.load('X11b_train.npz')
    S4b_test = np.load('S4b_test.npz')
    X11b_test = np.load('X11b_test.npz')

    train_data = np.concatenate((S4b_train['signal'], X11b_train['signal']), axis=0)
    train_label = np.concatenate((S4b_train['label'], X11b_train['label']), axis=0)
    test_data = np.concatenate((S4b_test['signal'], X11b_test['signal']), axis=0)
    test_label = np.concatenate((S4b_test['label'], X11b_test['label']), axis=0)


    train_label = train_label - 1
    test_label = test_label -1
    train_data = np.transpose(np.expand_dims(train_data, axis=1), (0, 1, 3, 2))
    test_data = np.transpose(np.expand_dims(test_data, axis=1), (0, 1, 3, 2))
   

    mask = np.where(np.isnan(train_data))
    train_data[mask] = np.nanmean(train_data)

    mask = np.where(np.isnan(test_data))
    test_data[mask] = np.nanmean(test_data)

   

    return train_data, train_label, test_data, test_label
# train_data, train_label, test_data, test_label = read_bci_data()
# eeg_data = test_data[0][0]
# num_channels, num_time_points = eeg_data.shape
# print(eeg_data)
# time_sequence = np.arange(num_time_points)
# fig, axes = plt.subplots(2, 1, figsize=(10, 6))

# # 繪製第一個通道的EEG數據
# axes[0].plot(time_sequence, eeg_data[0], label='Channel 1')
# axes[0].set_xlabel('Time')
# axes[0].set_ylabel('Amplitude')
# axes[0].set_title('EEG Data - Channel 1')
# axes[0].legend()

# # 繪製第二個通道的EEG數據
# axes[1].plot(time_sequence, eeg_data[1], label='Channel 2')
# axes[1].set_xlabel('Time')
# axes[1].set_ylabel('Amplitude')
# axes[1].set_title('EEG Data - Channel 2')
# axes[1].legend()

# # 調整子圖之間的間距
# plt.tight_layout()

# # 顯示圖形
# plt.show()
# import torch
# import torch.nn as nn
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN,self).__init__()#先找到CNN的父类（比如是类A），然后把类CNN的对象self转换为类A的对象，然后“被转换”的类A对象调用类A对象自己的__init__函数.
#         self.conv1 =nn.Sequential(
#             #卷积+激活+池化
#             #过滤器  高度-filter 用于提取出卷积出来的特征的属性
#             #图片的维度是 (1，28，28)1是chanel的维度，28x28的长宽高
#             nn.Conv2d(
#                 1, #in_channels=1图片的高度
#                 16,#out_channels=16 多少个输出的高度(filter的个数)
#                 5, #kernel_size=5，filter的高宽都是5个像素点
#                 1, #stride=1，卷积步长
#                 2, #padding=2填充，如果想输出的和输入的长宽一样则需要padding=(kernel_size-1)/2
#             ), #(16,28,28)
#             nn.ReLU(),
#             # 删选重要信息，参数(),kernel_size=2,把2x2的区域中选最大的值变成1x1 
#             nn.MaxPool2d(kernel_size=2),#(16,14,14)
#         )
#         #(16,14,14)
#         self.conv2=nn.Sequential(
#             nn.Conv2d(16,32,5,1,2),#(32,14,14)
#             nn.ReLU(),
#             #还可以用AvgPool3d，但一般用最大
#             nn.MaxPool2d(2),#(32,7,7)
#         )
#         #输出层 
#         self.out=nn.Linear(32*7*7,10)#（a，b）a是数据维度 b是分类器有十个
        
#     def forward(self,x):
#         x=self.conv1(x)
#         x=self.conv2(x)        #(batch,32,7,7)
#         x=x.view(x.size(0),-1) #(batch,32*7*7)
        
#         output =self.out(x)
        
#         return output
# cnn=CNN()
# print(cnn)

