import warnings     
warnings.filterwarnings('ignore')
import pathlib                                                                      
import numpy as np    
import pandas as pd
import matplotlib.pyplot as plt                  
import seaborn as sns  
import os       
import random
import splitfolders                               
from termcolor import colored                     

from tensorflow import keras 
import torch                  
import torch.nn as nn          # To work with Neural Networks
import torchvision 
import torch.nn.functional as F
import torch.utils.data

import torchvision.transforms as transforms  
from torchsummary import summary
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.io import read_image # For visualization
from torchvision.datasets import ImageFolder

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 62 * 62, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 62 * 62)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    # 保存每一轮训练时 训练数据 和 测试数据 的 loss 和 accuracy
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    for epoch in range(num_epochs):
        model.train() 
        running_loss = 0.0
        correct = 0
        total = 0
        # Training loop
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()               # Zero the parameter gradients
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total
        # Store training metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        # Validation phase
        model.eval() 
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():  # Disable gradient calculation for validation
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total
        # Store validation metrics
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        # Print epoch statistics
        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
    metrics_df = pd.DataFrame({
        'Epoch': range(1, num_epochs + 1),
        'Train Loss': train_losses,
        'Train Accuracy': train_accuracies,
        'Val Loss': val_losses,
        'Val Accuracy': val_accuracies
    })
    return metrics_df

if __name__ == "__main__":
    # 获取原始数据，并分组
    data_folder = '../../data/RiceImagesDataset'
    rice_classes = os.listdir(data_folder)
    rice_classes.remove('Rice_Citation_Request.txt')
    num_classes = len(rice_classes)
    splitfolders.ratio(data_folder, output='data_splitted' , seed=42, ratio=(0.75, 0.15, 0.1))
    data_dir = './data_splitted'
    data_dir = pathlib.Path(data_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 新建并加载模型
    model = CNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 1

    # 预处理数据
    Batch = 32
    transform = transforms.Compose([
            transforms.Resize((250,250)) ,
            transforms.ToTensor() ,
            transforms.Normalize((0),(1))])
    train_set = ImageFolder(data_dir.joinpath("train"), transform=transform)
    train_loader = DataLoader(train_set, batch_size=Batch, shuffle=True)
    val_set = ImageFolder(data_dir.joinpath("val"), transform=transform)
    val_loader = DataLoader(val_set, batch_size=Batch, shuffle = False)

    # 进行训练
    metrics_df = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)
    metrics_df
    torch.save(model.state_dict(), 'trained_model.pth')