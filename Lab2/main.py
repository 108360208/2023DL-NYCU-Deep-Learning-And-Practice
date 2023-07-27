from torch import Tensor, device, cuda, no_grad
from torch import max as tensor_max
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn   
import torch.optim as optim
from dataloader import read_bci_data
from torch import save
from torch.utils.data import TensorDataset, DataLoader
from functools import reduce
class EEGNet(nn.Module):
    def __init__(self, activation='leaky_relu'):    
        super(EEGNet,self).__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1 ,51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2,1), stride=(1,1), groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            activation_set(activation),
            nn.AvgPool2d(kernel_size=(1,4), stride=(1,4), padding=0),
            nn.Dropout(p=0.25)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1,15), stride=(1,1), padding=(0,7), bias=False),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            activation_set(activation),
            nn.AvgPool2d(kernel_size=(1,8), stride=(1,8), padding=0),
            nn.Dropout(p=0.25)
        )
        self.classify = nn.Sequential(
            nn.Linear(in_features=736, out_features=2, bias=True)
        )
    def forward(self,input):
        out = self.firstconv(input)
        out = self.depthwiseConv(out)
        out = self.separableConv(out)
        out_flattened = out.view(out.shape[0],-1)
        return self.classify(out_flattened)

class DeepConvNet(nn.Module):
    def __init__(self,filters_list,activation='leaky_relu'):
        assert len(filters_list)>0    
        super(DeepConvNet,self).__init__()
        self.layers = nn.ModuleList()
        self.firstconv = nn.Sequential(
            nn.Conv2d(1,filters_list[0],kernel_size=(1,5)),
            nn.Conv2d(filters_list[0],filters_list[0],kernel_size=(2,1)),
            nn.BatchNorm2d(filters_list[0],eps=1e-5, momentum=0.1),
            activation_set(activation),
            nn.MaxPool2d((1,2)),
            nn.Dropout(p=0.5)
        )

        for idx, num_filter in enumerate(filters_list[:-1],start=1):
            self.layers.append(nn.Sequential(
                nn.Conv2d(num_filter,filters_list[idx],kernel_size=(1,5)),
                nn.BatchNorm2d(filters_list[idx]),
                activation_set(activation),
                nn.MaxPool2d((1,2)),
                nn.Dropout(p=0.5)
            )) 
        flatten_size =  filters_list[-1] * reduce(
            lambda x,_: round((x-4)/2), filters_list    , 750)
        self.classify = nn.Sequential(
            nn.Linear(flatten_size, 2, bias=True),
        )

    def forward(self,input):
        out = self.firstconv(input)
        for layer in self.layers:
            out = layer(out)
        out = out.view(-1, self.classify[0].in_features)
        return self.classify(out)
def activation_set(activation):
    if(activation == "leeky_relu"):
        return nn.LeakyReLU(negative_slope=0.01)
    elif(activation == "relu"):
        return nn.ReLU()
    elif(activation == "elu"):
        return nn.ELU(alpha=0.1)

def train(net_type,activations,train_loader, test_loader, num_epochs=10, learning_rate=0.01, device='cuda'):

    best_models = {}   

    if device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    print(f'Using device: {device}')

    for activation in activations:
        train_accuracies = []
        eval_accuracies = [] 
        if(net_type == "eeg"):
            model = EEGNet(activation=activation)
        else:
            model = DeepConvNet(activation=activation,filters_list=[25,50,100,200])
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        best_accuracy = 0.0

        for epoch in range(num_epochs):
            #模型訓練
            model.train()
            correct = 0
            total = 0
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            if(epoch == 0 or (epoch+1) % 5 == 0):
                train_accuracies.append(100 * correct / total)
            #模型評估
            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model.forward(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
 
            accuracy = 100 * correct / total
            if(epoch == 0 or (epoch+1) % 5 == 0):
                eval_accuracies.append(accuracy)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Activation: {activation}, Test Accuracy: {accuracy:.2f}%')
            # 保存最佳模型的參數
            if (accuracy > best_accuracy):
                best_accuracy = accuracy
                best_model = model.state_dict()
        best_models[activation]={
            'best_model_state_dict': best_model,
            'best_accuracy': best_accuracy,
            'train_accuracies': train_accuracies,
            'eval_accuracies': eval_accuracies
        }
    filename = 'eeg_best_models.pth' if net_type == 'eeg' else 'deep_best_models.pth'
    # 保存模型參數到指定的檔案名稱
    save(best_models, filename)

        
def predict(net_type,loaded_models,test_loader,device="cuda"):

    if device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    for activation, model_key in loaded_models.items():
        if(net_type == "eeg"):
            model = EEGNet(activation=activation)
        else:
            model = DeepConvNet(activation=activation,filters_list=[25,50,100,200])
        model.load_state_dict(model_key['best_model_state_dict'])
        model.to(device)
        model.eval()
        print("The best model performance is below:")        
        # correct = 0
        # total = 0

        # with torch.no_grad():
        #     for inputs, labels in test_loader:
        #         inputs, labels = inputs.to(device), labels.to(device)
        #         outputs = model.forward(inputs)
        #         _, predicted = torch.max(outputs.data, 1)
        #         total += labels.size(0)
        #         correct += (predicted == labels).sum().item()

        accuracy = model_key['best_accuracy']
        print(f'Activation:{activation} Test Accuracy: {accuracy:.2f}%')
def plt_all_acc(loaded_models):
    import matplotlib.pyplot as plt
    all_accuracies = {}

# 遍歷 best_models 字典並繪製準確率曲線圖
    for activation, info in loaded_models.items():
        train_accuracies = info['train_accuracies']
        eval_accuracies = info['eval_accuracies']

        # 將激勵函數名稱作為 key，訓練和評估準確率作為 value，存入 all_accuracies 字典
        all_accuracies[activation] = {
            'train_accuracies': train_accuracies,
            'eval_accuracies': eval_accuracies
        }

    # 繪製所有激勵函數的訓練和評估準確率曲線圖
    plt.figure(figsize=(8, 6))

    for activation, accuracies in all_accuracies.items():
        train_accuracies = accuracies['train_accuracies']
        eval_accuracies = accuracies['eval_accuracies']
            # 將 x 軸的範圍遞增 10
        x_train = [5 * i for i in range(len(train_accuracies))]
        x_eval = [5 * i for i in range(len(eval_accuracies))]
        plt.plot(x_train, train_accuracies, label=f'Train Accuracy - {activation}')
        plt.plot(x_eval, eval_accuracies, label=f'Evaluation Accuracy - {activation}')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train and Evaluation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':

    train_data, train_label, test_data, test_label = read_bci_data()
    # (batch_size, channels, height, width)
    # print(train_data.shape)
    train_dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32), torch.tensor(train_label, dtype=torch.long)) 
    test_dataset = TensorDataset(torch.tensor(test_data, dtype=torch.float32),torch.tensor(test_label, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    activations = ["leeky_relu","relu","elu"]

    #train("deep",activations, train_loader, test_loader, num_epochs=1000)

    loaded_models = torch.load('eeg_best_models.pth')

    predict("eeg",loaded_models,test_loader)
    plt_all_acc(loaded_models)
    #print(EEGNet.parameters)
