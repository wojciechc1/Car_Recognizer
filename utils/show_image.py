import matplotlib.pyplot as plt
import torchvision
import torch

def show_batch(images, labels, class_names=None):
    # Odwrócenie normalizacji jeśli była, tutaj zakładamy brak
    grid = torchvision.utils.make_grid(images, nrow=8)
    np_img = grid.permute(1, 2, 0).numpy()  # [C,H,W] → [H,W,C]

    plt.figure(figsize=(12, 6))
    plt.imshow(np_img)
    plt.axis('off')

    # Dodanie labeli pod spodem (jeśli podano nazwy klas)
    if class_names:
        print("Labels:", [class_names[l.item()] for l in labels])
    else:
        print("Labels:", [l.item() for l in labels])

    plt.show()
