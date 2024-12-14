import os
# Disable albumentations update warning
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'  # Prevents albumentations from showing update warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from PIL import Image
import cv2
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

class Config:
    """Configuration for the model and training."""
    def __init__(self):
        # Basic configurations
        self.seed = 42             # Random seed for reproducibility
        self.image_size = 448      # Model's expected input size
        self.batch_size = 16       # Batch size for training (reduced for Kaggle's memory constraints)
        self.num_workers = 2       # Number of subprocesses for data loading (reduced for Kaggle)
        self.num_epochs = 15       # Total number of training epochs
        self.learning_rate = 1e-4  # Learning rate for the optimizer
        self.model_name = 'mobilenetv4_conv_aa_large.e230_r448_in12k_ft_in1k'       # Model architecture name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Device to run the model on
        
        # Kaggle specific paths
        self.data_dir = Path("/kaggle/input/rice-image-dataset/Rice_Image_Dataset") # Directory containing rice images
        self.output_dir = Path("/kaggle/working")                                   # Directory to save outputs
        self.model_path = self.output_dir / "best_model.pth"                        # Path to save the best model
        
        # Categories
        self.categories = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']  # List of rice categories
        self.num_classes = len(self.categories)                                     # Number of distinct classes
        
        # Normalization values
        self.mean = [0.485, 0.456, 0.406]  # Mean for image normalization (ImageNet standards)
        self.std = [0.229, 0.224, 0.225]   # Standard deviation for image normalization (ImageNet standards)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)     # Ensure the output directory exists

def plot_samples(config: Config, save_path: str = None):
    """Plot sample images from each category."""
    fig, ax = plt.subplots(ncols=5, figsize=(20,5))     # Create a figure with 5 subplots
    fig.suptitle('Rice Categories')                     # Set the overall title
    
    for idx, category in enumerate(config.categories):
        # Get the first image in the category
        image_path = next(iter(list((config.data_dir / category).glob('*.jpg'))))
        image = cv2.imread(str(image_path))             # Read the image using OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        ax[idx].set_title(category)  # Set the title of the subplot to the category name
        ax[idx].imshow(image)        # Display the image
        ax[idx].axis('off')          # Hide the axis
    
    if save_path:
        plt.savefig(save_path)  # Save the figure if a save path is provided
    plt.close()  # Close the figure to free up memory

class RiceDataset(Dataset):
    """Dataset class for rice images."""
    def __init__(self, images: List[Path], labels: List[int], transform=None):
        self.images = images        # List of image file paths
        self.labels = labels        # Corresponding list of labels
        self.transform = transform  # Data augmentation and preprocessing transforms
        
    def __len__(self) -> int:
        return len(self.images)  # Return the total number of samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path = str(self.images[idx])  # Get the image path as a string
        try:
            image = cv2.imread(image_path)                  # Read the image using OpenCV
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")  # Print error message if image fails to load
            image = np.zeros((448, 448, 3), dtype=np.uint8)       # Use a blank image in case of error
        
        if self.transform:
            augmented = self.transform(image=image)  # Apply transformations if any
            image = augmented['image']               # Get the transformed image
        
        label = self.labels[idx]  # Get the corresponding label
        return image, label       # Return the image and label

def get_transforms(config: Config, is_train: bool = False):
    """Get image transforms for training/validation."""
    if is_train:
        return A.Compose([
            A.RandomResizedCrop(
                height=config.image_size,
                width=config.image_size,
                scale=(0.8, 1.0)
            ),  # Randomly crop and resize the image
            A.HorizontalFlip(p=0.5),  # Random horizontal flip with 50% probability
            A.VerticalFlip(p=0.5),    # Random vertical flip with 50% probability
            A.ShiftScaleRotate(
                shift_limit=0.2,
                scale_limit=0.2,
                rotate_limit=30,
                p=0.5
            ),  # Randomly shift, scale, and rotate the image
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5
            ),  # Randomly change hue, saturation, and value
            A.RandomBrightnessContrast(p=0.5),               # Randomly adjust brightness and contrast
            A.Normalize(mean=config.mean, std=config.std),   # Normalize the image
            ToTensorV2()                                     # Convert image to PyTorch tensor
        ])
    else:
        return A.Compose([
            A.Resize(config.image_size, config.image_size),  # Resize the image to the specified size
            A.Normalize(mean=config.mean, std=config.std),   # Normalize the image
            ToTensorV2()                                     # Convert image to PyTorch tensor
        ])

def prepare_data(config: Config) -> Tuple[List, List]:
    """Prepare data for training."""
    images, labels = [], []  # Initialize lists to store image paths and labels
    label_dict = {cat: idx for idx, cat in enumerate(config.categories)}  # Create a mapping from category to index
    
    print("Loading data...")
    for category in config.categories:
        category_path = config.data_dir / category                # Path to the current category's images
        image_paths = list(category_path.glob('*.jpg'))[:600]     # Get up to 600 images per category
        print(f"{category}: {len(image_paths)} images")           # Print the number of images loaded for the category
        images.extend(image_paths)                                # Add image paths to the list
        labels.extend([label_dict[category]] * len(image_paths))  # Add corresponding labels
    
    return images, labels  # Return the lists of image paths and labels

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> float:
    """Train for one epoch."""
    model.train()   # Set the model to training mode
    total_loss = 0  # Initialize total loss for the epoch

    # Progress bar for training
    with tqdm(train_loader, desc=f'Epoch {epoch + 1} - Training') as pbar:  
        for images, labels in pbar:
            images = images.to(device)         # Move images to the specified device
            labels = labels.to(device)         # Move labels to the device
            
            optimizer.zero_grad()              # Reset gradients
            outputs = model(images)            # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            
            loss.backward()                    # Backward pass
            optimizer.step()                   # Update model parameters
            
            total_loss += loss.item()          # Accumulate loss
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})  # Update progress bar with current loss

    # Return average loss for the epoch: ceil(total_number_of_samples / batch_size)
    return total_loss / len(train_loader)

@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = 'Validating'
) -> Tuple[float, float]:
    """Validate the model."""
    model.eval()    # Set the model to evaluation mode
    total_loss = 0  # Initialize total loss for validation
    correct = 0     # Initialize count of correct predictions
    total = 0       # Initialize total number of samples
    
    with tqdm(val_loader, desc=desc) as pbar:              # Progress bar for validation
        for images, labels in pbar:            
            images = images.to(device)                     # Move images to the device
            labels = labels.to(device)                     # Move labels to the device
                        
            outputs = model(images)                        # Forward pass
            loss = criterion(outputs, labels)              # Compute loss
                        
            total_loss += loss.item()                      # Accumulate loss
            _, predicted = torch.max(outputs.data, 1)      # Get predicted class
            total += labels.size(0)                        # Update total count
            correct += (predicted == labels).sum().item()  # Update correct predictions count
            
            accuracy = 100 * correct / total  # Calculate accuracy
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'accuracy': f'{accuracy:.2f}%'
            })  # Update progress bar with current loss and accuracy
    
    accuracy = 100 * correct / total               # Final accuracy calculation
    return total_loss / len(val_loader), accuracy  # Return average loss and accuracy

def main():
    """Main function to execute the training and evaluation pipeline."""
    config = Config()  # Initialize configuration
    print(f"Using device: {config.device}")  # Print the device being used
    
    # Set random seeds for reproducibility
    torch.manual_seed(config.seed)  # Set PyTorch seed
    np.random.seed(config.seed)     # Set NumPy seed
    
    # Plot and save sample images from each category
    plot_samples(config, save_path=str(config.output_dir / "samples.png"))  # Save sample images to output directory
    
    # Prepare data by loading image paths and labels
    images, labels = prepare_data(config)  # Get lists of image paths and labels
    
    # Split data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels, test_size=0.3, random_state=config.seed
    )  # Split into 70% train and 30% temporary
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=config.seed
    )  # Split temporary into 15% validation and 15% test
    
    print(f"\nDataset splits:")  # Print dataset split information
    print(f"Train: {len(X_train)} images")
    print(f"Val: {len(X_val)} images")
    print(f"Test: {len(X_test)} images")
    
    # Create datasets with appropriate transforms
    train_dataset = RiceDataset(
        X_train, y_train,
        transform=get_transforms(config, is_train=True)   # Apply training transforms
    )
    val_dataset = RiceDataset(
        X_val, y_val,
        transform=get_transforms(config, is_train=False)  # Apply validation transforms
    )
    test_dataset = RiceDataset(
        X_test, y_test,
        transform=get_transforms(config, is_train=False)  # Apply test transforms
    )
    
    # Create data loaders for each dataset
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,   # Batch size from config
        shuffle=True,                   # Shuffle training data
        num_workers=config.num_workers  # Number of workers from config
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,   # Batch size from config
        shuffle=False,                  # Do not shuffle validation data
        num_workers=config.num_workers  # Number of workers from config
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,   # Batch size from config
        shuffle=False,                  # Do not shuffle test data
        num_workers=config.num_workers  # Number of workers from config
    )
    
    # Initialize the model using the timm library
    model = timm.create_model(
        config.model_name,              # Model architecture name
        pretrained=True,                # Use pretrained weights
        num_classes=config.num_classes  # Number of output classes
    ).to(config.device)                 # Move the model to the specified device
    
    criterion = nn.CrossEntropyLoss()   # Define the loss function
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)  # Initialize the optimizer with learning rate
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs  # Cosine annealing scheduler over the number of epochs
    )
    
    # Initialize variables to track the best validation accuracy
    best_val_acc = 0
    
    # Training loop over epochs
    for epoch in range(config.num_epochs):
        # Train for one epoch and get the average training loss
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, config.device, epoch
        )
        
        # Validate the model and get the average validation loss and accuracy
        val_loss, val_acc = validate(
            model, val_loader, criterion, config.device,
            desc=f'Epoch {epoch + 1} - Validating'
        )
        
        scheduler.step()  # Update the learning rate scheduler
        
        # Print training and validation metrics
        print(f'\nEpoch {epoch + 1}:')
        print(f'Train Loss: {train_loss:.4f}')  # Print average training loss
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%')  # Print average validation loss and accuracy
        
        # Save the model if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc  # Update the best validation accuracy
            torch.save(model.state_dict(), config.model_path)  # Save the model weights
            print(f'Saved best model with validation accuracy: {val_acc:.2f}%')  # Inform about the saved model
    
    # Load the best model for evaluation on the test set
    print("\nEvaluating best model on test set...")
    model.load_state_dict(torch.load(config.model_path))  # Load the best model weights
    test_loss, test_acc = validate(
        model, test_loader, criterion, config.device,
        desc='Testing'
    )
    print(f'\nTest Accuracy: {test_acc:.2f}%')  # Print test accuracy
    
    # Generate and print a classification report
    model.eval()     # Set the model to evaluation mode
    all_preds = []   # List to store all predictions
    all_labels = []  # List to store all true labels
    
    print("\nGenerating classification report...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Predicting'):
            images = images.to(config.device)          # Move images to the device
            outputs = model(images)                    # Forward pass
            _, predicted = torch.max(outputs.data, 1)  # Get predicted classes
            all_preds.extend(predicted.cpu().numpy())  # Append predictions to the list
            all_labels.extend(labels.numpy())          # Append true labels to the list
    
    # Create a classification report using sklearn
    report = classification_report(
        all_labels,
        all_preds,
        target_names=config.categories,
        digits=4
    )
    
    print('\nClassification Report:')  # Header for the classification report
    print(report)  # Print the classification report
    
    # Save the classification report to a text file
    with open(config.output_dir / 'classification_report.txt', 'w') as f:
        f.write(report)  # Write the report to a file

if __name__ == '__main__':
    main()  # Execute the main function when the script is run