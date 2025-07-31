import torch

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
        total += labels.size(0)

        if batch_idx >= max_batches:
            break

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total if total > 0 else 0

    print(f"Train: avg_Loss: {avg_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")
    return avg_loss, accuracy