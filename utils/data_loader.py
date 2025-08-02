import scipy.io as sio
import pandas as pd

import os

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from torch.utils.data import DataLoader


# Ścieżka do folderu ze zdjęciami
train_image_dir = "../Datasets/cars_ds/cars_train/cars_train"
test_image_dir = "../Datasets/cars_ds/cars_test/cars_test"

# sciezka do pliku z adnotacjami
train_anno_path = "../Datasets/cars_ds/car_devkit/devkit/cars_train_annos.mat"
test_anno_path = "../Datasets/cars_ds/car_devkit/devkit/cars_test_annos.mat"

# Wczytaj dane
train_annos = sio.loadmat(train_anno_path)['annotations']
test_annos = sio.loadmat(train_anno_path)['annotations']

# class names
meta_path = "../Datasets/cars_ds/car_devkit/devkit/cars_meta.mat"


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def get_class_names(meta_path = meta_path):
    meta = sio.loadmat(meta_path)
    raw_class_names = meta['class_names'][0]
    class_names = [str(c.squeeze()) for c in raw_class_names]
    model_to_brand = [ name.split()[0] for i, name in enumerate(class_names)]

    unique_brands = list(dict.fromkeys(model_to_brand))  # ['AM', 'Acura', 'Aston']

    # słownik id -> brand
    brand_keys = [brand for brand in unique_brands]

    # listę oryginalnych marek na id
    brands_idx = [brand_keys.index(brand) for brand in model_to_brand]

    class_names = {id: [model, brands_idx[id]] for id, model in enumerate(class_names)}
    return class_names, brand_keys


def parse_annotations(annos):
    data = []
    for a in annos[0]:
        item = {
            "file": a["fname"][0],
            "class": int(a["class"][0][0]),
            "bbox_x1": int(a["bbox_x1"][0][0]),
            "bbox_y1": int(a["bbox_y1"][0][0]),
            "bbox_x2": int(a["bbox_x2"][0][0]),
            "bbox_y2": int(a["bbox_y2"][0][0]),
        }
        data.append(item)
    return pd.DataFrame(data)


class CarsDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.class_names, self.brand_keys = get_class_names(meta_path=meta_path)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = Image.open(row['file_path']).convert("RGB")
        label = self.class_names[row['class'] - 1][1]  # klasy są od 1 do 196, więc trzeba -1

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_brand_keys(self):
        return self.brand_keys

def get_data_loaders(
        train_image_dir = train_image_dir,
        train_annos = train_annos,
        test_image_dir = test_image_dir,
        test_annos = test_annos,
        transform = transform,
        batch_size = 32,):

    train_df = parse_annotations(train_annos)
    train_df['file_path'] = train_df['file'].apply(lambda x: os.path.join(train_image_dir, x))

    test_df = parse_annotations(test_annos)
    test_df['file_path'] = test_df['file'].apply(lambda x: os.path.join(test_image_dir, x))

    train_dataset = CarsDataset(train_df, transform=train_transform)
    test_dataset = CarsDataset(test_df, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset.get_brand_keys()


