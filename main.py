


'''from utils.plot_metrics import plot_metrics

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18, ResNet18_Weights

from scripts.train_context import raw
from scripts.test_context import test

from torch.utils.data import Subset

import warnings

warnings.filterwarnings("ignore", message="image file could not be identified because AVIF support not installed")



if __name__ == "__main__":
    from utils.dataset import SafeImageFolder, safe_loader, train_transform, val_transform  # jeśli masz osobny plik

    full_dataset = SafeImageFolder("data/dataset/raw", transform=None)

    # Podział na indeksy
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Przypisz transformacje ręcznie
    train_dataset = Subset(SafeImageFolder("data/dataset/raw", transform=train_transform, loader=safe_loader), train_dataset.indices)
    val_dataset = Subset(SafeImageFolder("data/dataset/raw", transform=val_transform, loader=safe_loader), val_dataset.indices)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    #train_dataset = SafeImageFolder("data/dataset/raw", transform=train_transform, loader=safe_loader)
    #test_dataset = SafeImageFolder("data/dataset/test", transform=transform)

    #train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    #test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = ResNet18_Weights.DEFAULT  # najnowsze dostępne wagi
    model = resnet18(weights=weights)


    model.fc = nn.Linear(model.fc.in_features, 4)


    for param in model.parameters():
        param.requires_grad = False

    # Odmrożenie ostatniego bloku ResNet18 (layer4) oraz klasyfikatora
    for param in model.layer3.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True


    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.fc.parameters(), lr=0.01)

    #scheduler = StepLR(optimizer, step_size=2, gamma=0.33)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    # metrics
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []


    for epoch in range(20):

        train_loss, train_acc = raw(train_loader, model, criterion, optimizer, device)
        test_loss, test_acc = test(val_loader, model, criterion, device)
        scheduler.step(test_loss)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
    
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        #TODO - lepsza walidacja, odblokowac warstwy, val set i sheduler lepszy, dataset wiekszy


    plot_metrics(train_losses, test_losses, train_accs, test_accs)
'''


from pipelines.context_pipeline import ContextPipeline

if __name__ == "__main__":
    pipeline = ContextPipeline("weights/context_classifier.pth")
    label, probs = pipeline.run("example.jpg")
    print(label)