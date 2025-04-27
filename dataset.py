import os
import random
import shutil
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import batch_size, resize_x, resize_y

# Define transformations
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(resize_x),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
    transforms.RandomAffine(20, translate=(0.1, 0.1), shear=10),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(resize_x),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Paths
train_dir = "path/to/working/train"
val_dir = "path/to/working/val"
test_dir = "path/to/working/test"

# Custom dataset loader (not necessary to redefine ImageFolder)
FlowerDataset = datasets.ImageFolder

# Loaders
train_loader = DataLoader(FlowerDataset(train_dir, transform=train_transforms), batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(FlowerDataset(val_dir, transform=val_test_transforms), batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(FlowerDataset(test_dir, transform=val_test_transforms), batch_size=batch_size, shuffle=False, num_workers=2)
