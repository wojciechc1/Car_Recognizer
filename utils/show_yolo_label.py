import cv2
import matplotlib.pyplot as plt

def show_yolo_label(img_path, label_path):
    # Wczytaj obraz
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    # Wczytaj etykiety YOLO
    with open(label_path, 'r') as f:
        lines = f.readlines()

    # Rysuj bounding boxy
    for line in lines:
        cls, x_c, y_c, bw, bh = map(float, line.strip().split())
        # Zamień YOLO na pixele
        x_c *= w
        y_c *= h
        bw *= w
        bh *= h

        x1 = int(x_c - bw / 2)
        y1 = int(y_c - bh / 2)
        x2 = int(x_c + bw / 2)
        y2 = int(y_c + bh / 2)

        # Rysuj prostokąt
        cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
        cv2.putText(img, f"{int(cls)}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Pokaż
    plt.imshow(img)
    plt.axis('off')
    plt.show()


import os

def rename_images_labels(folder_path, start_index=100):
    # Pobierz listę plików jpg i txt w folderze
    images = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
    labels = sorted([f for f in os.listdir(folder_path) if f.endswith('.txt')])

    # Zakładam, że pliki są parami image_0.jpg <-> image_153.txt
    for i, (img, lbl) in enumerate(zip(images, labels)):
        # Nowa nazwa z przesunięciem
        new_index = i + start_index
        new_img_name = f"image_{new_index}.jpg"
        new_lbl_name = f"image_{new_index}.txt"

        # Pełne ścieżki
        img_path = os.path.join(folder_path, img)
        lbl_path = os.path.join(folder_path, lbl)

        new_img_path = os.path.join(folder_path, new_img_name)
        new_lbl_path = os.path.join(folder_path, new_lbl_name)

        # Zmień nazwę plików
        os.rename(img_path, new_img_path)
        os.rename(lbl_path, new_lbl_path)

    print(f"Zmieniono nazwy {len(images)} par plików, zaczynając od indexu {start_index}.")



#rename_images_labels('../data/dataset_logo_bb/raw', start_index=100)
show_yolo_label('../data/dataset_logo_bb/yolo_ds/images/train/image_126.jpg',
                '../data/dataset_logo_bb/yolo_ds/labels/train/image_126.txt')