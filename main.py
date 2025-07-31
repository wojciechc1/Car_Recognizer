from time import sleep

from utils.data_loader import get_data_loaders
from utils.show_image import show_batch

import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim


train_loader, test_loader, brand_keys = get_data_loaders(batch_size=1)


# Wczytanie ResNet18 z pretrained wagami
model = models.resnet18(pretrained=True)

num_classes = 196  # lub ile masz klas

# Zamieniamy ostatnią warstwę FC (fully connected)
model.fc = nn.Linear(model.fc.in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

for epoch in range(3):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print(loss.item())
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")




        #show_batch(img, label, brand_keys)
