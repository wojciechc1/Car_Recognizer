import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.models import resnet18, ResNet18_Weights
from utils.dataset import SafeImageFolder, train_transform, val_transform, safe_loader
from scripts.test_context import test
from utils.plot_metrics import plot_metrics


def train(train_loader, model, criterion, optimizer, device, max_batches=6):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print(loss.item())
        total_loss += loss.item()

        # Oblicz accuracy
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        #print("shape outputs:", outputs.shape)
        #print(labels[0], preds[0], outputs[0])
        #print(labels.min(), labels.max(), labels.dtype)
        #print("shape outputs", outputs.shape, " shape labels", labels.shape)
        total += labels.size(0)


    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total if total > 0 else 0

    print(f"Train: avg_Loss: {avg_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")
    return avg_loss, accuracy


def train_context_classifier():
    full_dataset = SafeImageFolder("data/dataset/train", transform=None)

    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_dataset = Subset(SafeImageFolder("data/dataset/train", transform=train_transform, loader=safe_loader), train_dataset.indices)
    val_dataset = Subset(SafeImageFolder("data/dataset/train", transform=val_transform, loader=safe_loader), val_dataset.indices)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 4)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    train_losses, test_losses, train_accs, test_accs = [], [], [], []

    for epoch in range(20):
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, device)
        test_loss, test_acc = test(val_loader, model, criterion, device)
        scheduler.step(test_loss)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}, Test Acc: {test_acc:.2f}")

    plot_metrics(train_losses, test_losses, train_accs, test_accs)
    torch.save(model.state_dict(), "weights/context_classifier.pth")

if __name__ == "__main__":
    train_context_classifier()