import os


def count_files(folder, exts=['.jpg', '.jpeg', '.png']):
    return len([f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in exts])

def count_label_files(folder):
    return len([f for f in os.listdir(folder) if f.endswith('.txt')])

def check_data_integrity(image_folder, label_folder):
    images = [os.path.splitext(f)[0] for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    labels = [os.path.splitext(f)[0] for f in os.listdir(label_folder) if f.endswith('.txt')]

    missing_labels = set(images) - set(labels)
    missing_images = set(labels) - set(images)

    if missing_labels:
        print(f"Brakujące labelki dla obrazów: {missing_labels}")
    if missing_images:
        print(f"Brakujące obrazy dla labeli: {missing_images}")
    if not missing_labels and not missing_images:
        print("Liczba obrazów i labeli się zgadza!")

# Ścieżki - dostosuj do swojego projektu
train_images = '../data/dataset_logo_bb/yolo_ds/images/train'
train_labels = '../data/dataset_logo_bb/yolo_ds/labels/train'
val_images = '../data/dataset_logo_bb/yolo_ds/images/val'
val_labels = '../data/dataset_logo_bb/yolo_ds/labels/val'

print("===== Sprawdzam dane treningowe =====")
print(f"Liczba obrazów treningowych: {count_files(train_images)}")
print(f"Liczba labeli treningowych: {count_label_files(train_labels)}")
check_data_integrity(train_images, train_labels)

print("\n===== Sprawdzam dane walidacyjne =====")
print(f"Liczba obrazów walidacyjnych: {count_files(val_images)}")
print(f"Liczba labeli walidacyjnych: {count_label_files(val_labels)}")
check_data_integrity(val_images, val_labels)