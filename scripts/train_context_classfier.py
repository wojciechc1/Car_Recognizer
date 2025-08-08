import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from utils.dataset import train_transform, val_transform
from utils.plot_metrics import plot_metrics

from train import train
from test import test




def train_context_classifier():

    # Dataset
    train_dataset = datasets.ImageFolder(root='../../Datasets/data/dataset_masked/train', transform=train_transform)
    test_dataset = datasets.ImageFolder(root='../../Datasets/data/dataset_masked/test', transform=val_transform)

    # DataLoadery
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Sprawdzenie klas
    print("Classes:", train_dataset.classes)

    print(len(train_loader))
    print(len(test_loader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True
    params_to_update = list(model.fc.parameters()) + list(model.layer4.parameters())

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params_to_update, lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    train_losses, test_losses, train_accs, test_accs = [], [], [], []

    for epoch in range(10):
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, device)
        test_loss, test_acc = test(test_loader, model, criterion, device)
        scheduler.step(test_loss)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}, Test Acc: {test_acc:.2f}")

    plot_metrics(train_losses, test_losses, train_accs, test_accs)
    torch.save(model.state_dict(), "../config/context_classifier.pth")

if __name__ == "__main__":
    train_context_classifier()