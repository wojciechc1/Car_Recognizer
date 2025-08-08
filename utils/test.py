import torch

def test(test_loader, model, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        # print(loss.item())
        total_loss += loss.item()

        # Oblicz accuracy
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)


    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total if total > 0 else 0

    print(f"Val: avg_Loss: {avg_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")
    return avg_loss, accuracy
