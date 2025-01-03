import warnings     
warnings.filterwarnings('ignore')
import pathlib                 
import pandas as pd
import os       
import splitfolders
import torch
import torch.nn as nn
import torchvision.transforms as transforms  
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from model import CNN

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    # 保存每一轮训练时 训练数据 和 测试数据 的 loss 和 accuracy
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    for epoch in range(num_epochs):
        model.train() # 设置模型为训练模式
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
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
    torch.save(model.state_dict(), 'trained_model.pth')