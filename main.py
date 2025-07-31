from time import sleep

from utils.data_loader import get_data_loaders
from utils.show_image import show_batch
from utils.plot_metrics import plot_metrics

import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

from train import train
from test import test


train_loader, test_loader, brand_keys = get_data_loaders(batch_size=32)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=True)

num_classes = len(brand_keys)

# Zamieniamy ostatnią warstwę FC (fully connected)
model.fc = nn.Linear(model.fc.in_features, num_classes)

model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# metrics
train_losses = []
test_losses = []
train_accs = []
test_accs = []


max_batches = 6

for epoch in range(6):
    train_loss, train_acc = train(train_loader, model, criterion, optimizer, device, max_batches=10)
    test_loss, test_acc = test(test_loader, model, criterion, device, max_batches=10)

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    train_accs.append(test_acc)
    test_accs.append(test_acc)


plot_metrics(train_losses, test_losses, train_accs, test_accs)

        #show_batch(img, label, brand_keys)
