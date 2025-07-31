from utils.data_loader import get_data_loaders, get_class_names
from utils.show_image import show_batch

train_loader, test_loader = get_data_loaders(batch_size=2)

class_names = get_class_names()


for img, label in train_loader:
    print(img.shape)
    print(label.shape)
    show_batch(img, label, class_names)
