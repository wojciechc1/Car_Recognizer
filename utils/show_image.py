import torchvision
import torch
import matplotlib.pyplot as plt
import numpy as np

# Funkcja do odnormalizowania obrazu
def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean

def show_batch(images, labels, brand_keys, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # ODNORMALIZUJ każdy obraz w batchu
    denorm_images = torch.stack([denormalize(img, mean, std) for img in images])

    # Tworzenie siatki obrazów
    grid = torchvision.utils.make_grid(denorm_images, nrow=8)
    np_img = grid.permute(1, 2, 0).numpy()  # [C,H,W] → [H,W,C]

    plt.figure(figsize=(12, 6))
    plt.imshow(np.clip(np_img, 0, 1))  # upewniamy się, że wartości są w [0,1]
    plt.axis('off')

    print([brand_keys[l.item()] for l in labels])
    plt.show()
