from time import sleep

from utils.data_loader import get_data_loaders
from utils.show_image import show_batch

train_loader, test_loader, brand_keys = get_data_loaders(batch_size=1)



for img, label in train_loader:
    print(img.shape)
    print(label.shape)
    show_batch(img, label, brand_keys)
