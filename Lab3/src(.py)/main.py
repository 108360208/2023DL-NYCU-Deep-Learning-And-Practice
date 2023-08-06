import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import dataloader
import ResNet
import torch.optim as optim
import torch.nn as nn
import torch
from torch import save
import copy
import os

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_accuracy = correct_predictions.double() / len(train_loader.dataset)
    return epoch_loss, epoch_accuracy

# 定义评估函数
def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_accuracy = correct_predictions.double() / len(test_loader.dataset)
    return epoch_loss, epoch_accuracy

def save_result(csv_path, predict_result):
    df = pd.read_csv(csv_path)
    new_df = pd.DataFrame()
    new_df['ID'] = df['Path']
    new_df["label"] = predict_result
    new_df.to_csv("./resnet18.csv", index=False)

def test(model, test_loader, device):
    model.eval()
    predict_result=[]
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predict_result.append(preds)
    predict_result = torch.cat(predict_result)

    return predict_result
#只需要更改net_type,model 
net_type = "resnet18"
model = ResNet.ResNet18(num_classes=2)
#print(model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

batch_size = 32
learning_rate = 0.1
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9, dampening=0,weight_decay = 0.001)
optimizer = optim.Adamax(model.parameters(), lr=learning_rate)
current_directory = os.getcwd()
file_path = os.path.join(current_directory, "net_weights", net_type)

#print(current_directory)
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    root = "new_dataset"
    transform_dict = dataloader.transform(net_type)
    #data loader 
    Leukemia_dataset = dataloader.LeukemiaLoader(root,"train",transform = transform_dict["train"])
    train_loader =  DataLoader(Leukemia_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    Leukemia_dataset = dataloader.LeukemiaLoader(root,"valid",transform = transform_dict["valid"])
    test_loader =  DataLoader(Leukemia_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    test_loader =  DataLoader(Leukemia_dataset, batch_size=batch_size, shuffle=False)
    best_model = torch.load(os.path.join(file_path,"best_models.pth"))
    best_model_state_dict = best_model['best_model_state_dict']
    print(best_model["eval_accuracies"])
    model.load_state_dict(best_model_state_dict)
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    print(test_loss,test_accuracy)
    # #第一次訓練
    # num_epochs = 30
    # best_accuracy = 0.0
    # train_accuracies = []
    # eval_accuracies = [] 

    # 再練一次
    # num_epochs = 50
    # last_model = torch.load("C:/Users/Steven/Desktop/課程資料/交大/2023DL/Lab/Lab3/Lab3-Leukemia_Classification/test_weight/last_models.pth")
    # model.load_state_dict(last_model['last_model_state_dict'])
    # optimizer.load_state_dict(last_model["optimizer_state_dict"])
    # best_accuracy = max(last_model['eval_accuracies'])
    # print(best_accuracy)
    # train_accuracies = last_model['train_accuracies']
    # eval_accuracies = last_model['eval_accuracies']

    # for epoch in range(num_epochs):
    #     # 训练
    #     train_loss, train_accuracy = train(model, train_loader, optimizer, criterion, device)
    #     train_accuracies.append(train_accuracy)
    #     print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")
        
    #     # 评估
    #     test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    #     eval_accuracies.append(test_accuracy)  
    #     print(f"Epoch {epoch+1}/{num_epochs}, Testing Loss: {test_loss:.4f}, Testing Accuracy: {test_accuracy:.4f}")
        
    #     if (test_accuracy > best_accuracy):
    #         best_accuracy = test_accuracy
    #         best_model = copy.deepcopy(model.state_dict())
    #         best_optimizer = copy.deepcopy(optimizer.state_dict())
    #     torch.cuda.empty_cache()
    # last_models={
    #         'last_model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'loss': test_loss,
    #         'train_accuracies': train_accuracies,
    #         'eval_accuracies': eval_accuracies
    #     }
    # save(last_models,os.path.join(file_path,"last_models.pth"))
    # #save the best model weights
    # best_models={
    #         'best_model_state_dict': best_model,
    #         'optimizer_state_dict': best_optimizer,
    #         'loss': test_loss,
    #         'train_accuracies': train_accuracies,
    #         'eval_accuracies': eval_accuracies
    #     }
    # save(best_models,os.path.join(file_path,"best_models.pth"))

    #save the last train model weights 
    # Leukemia_dataset = dataloader.LeukemiaLoader(root,"test",transform=data_transform)
    # test_loader =  DataLoader(Leukemia_dataset, batch_size=batch_size, shuffle=False)
    # best_model = torch.load("C:\\Users\\Steven\\Desktop\\課程資料\\交大\\2023DL\\Lab\\Lab3\\Lab3-Leukemia_Classification\\test_weight\\18best.pth")
    # best_model_state_dict = best_model['best_model_state_dict']
    # model.load_state_dict(best_model_state_dict)
    # test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    # print(test_loss, test_accuracy)
    # predict_result = test(model,test_loader, device).cpu().numpy()
    # save_result("C:\\Users\\Steven\\Desktop\\課程資料\\交大\\2023DL\\Lab\\Lab3\\resnet_18_test.csv",predict_result=predict_result)