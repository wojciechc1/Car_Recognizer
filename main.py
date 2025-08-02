from time import sleep

from utils.show_image import show_batch
from utils.plot_metrics import plot_metrics

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights

from train import train
from test import test

import warnings

warnings.filterwarnings("ignore", message="image file could not be identified because AVIF support not installed")



if __name__ == "__main__":
    from utils.dataset import SafeImageFolder, safe_loader, train_transform  # jeśli masz osobny plik


    train_dataset = SafeImageFolder("data/dataset/train", transform=train_transform, loader=safe_loader)
    #test_dataset = SafeImageFolder("data/dataset/test", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    #test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = ResNet18_Weights.DEFAULT  # najnowsze dostępne wagi
    model = resnet18(weights=weights)


    model.fc = nn.Linear(model.fc.in_features, 4)


    for param in model.parameters():
        param.requires_grad = False

    # Odmrożenie ostatniego bloku ResNet18 (layer4) oraz klasyfikatora
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True


    model.to(device)

    criterion = nn.CrossEntropyLoss()

    from torch.optim.lr_scheduler import StepLR
    optimizer = optim.Adam(model.fc.parameters(), lr=0.01)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.33)

    # metrics
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []


    for epoch in range(10):

        train_loss, train_acc = train(train_loader, model, criterion, optimizer, device)
        #test_loss, test_acc = test(test_loader, model, criterion, device, max_batches=10)
        scheduler.step()

        train_losses.append(train_loss)
        test_losses.append(0)
    
        train_accs.append(train_acc)
        test_accs.append(0)


    plot_metrics(train_losses, test_losses, train_accs, test_accs)


