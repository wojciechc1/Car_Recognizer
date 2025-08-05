import os
import xml.etree.ElementTree as ET
from pathlib import Path
import re

# MAPA klas – dostosuj do swoich danych
CLASS_MAP = {
    "audi": 0,
    "bmw": 1,
    "mercedes": 2,
    "toyota": 3,
    # ...
}

def convert_annotation(xml_file: Path, output_dir: Path):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    try:
        img_w = int(root.find('size/width').text)
        img_h = int(root.find('size/height').text)
        if img_w == 0 or img_h == 0:
            print(f"[ERROR] Rozmiar obrazu 0 w {xml_file.name}, pominięto.")
            return
    except:
        print(f"[ERROR] Brak rozmiaru obrazu w {xml_file.name}, pominięto.")
        return

    lines = []

    for obj in root.findall('object'):
        name = obj.find('name').text.lower()

        if name not in CLASS_MAP:
            print(f"[WARN] Pominięto nieznaną klasę: {name}")
            continue

        class_id = CLASS_MAP[name]
        bndbox = obj.find('bndbox')

        try:
            xmin = int(bndbox.find('xmin').text)
            xmax = int(bndbox.find('xmax').text)
            ymin = int(bndbox.find('ymin').text)
            ymax = int(bndbox.find('ymax').text)
        except:
            print(f"[ERROR] Błąd w bbox w {xml_file.name}, pominięto.")
            continue

        x_center = ((xmin + xmax) / 2) / img_w
        y_center = ((ymin + ymax) / 2) / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h

        line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        lines.append(line)

    output_file = output_dir / (xml_file.stem + ".txt")
    with open(output_file, "w") as f:
        f.write("\n".join(lines))


def natural_sort_key(path: Path):
    # Sortowanie typu: image_1, image_2, ..., image_10
    return [int(t) if t.isdigit() else t.lower() for t in re.split('([0-9]+)', path.stem)]


def convert_folder(input_dir: str, output_dir: str):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(input_path.glob("*.xml"), key=natural_sort_key)
    print(f"[INFO] Znaleziono {len(xml_files)} plików XML")

    for xml_file in xml_files:
        convert_annotation(xml_file, output_path)

    print(f"[DONE] Konwersja zakończona. Wynik w: {output_dir}")


# PRZYKŁADOWE UŻYCIE:
if __name__ == "__main__":
    input_dir = "../data/dataset_logo_bb/raw/"  # np. "annotations/"
    output_dir = "../data/dataset_logo_bb/yolo/"  # np. "labels/"

    convert_folder(input_dir, output_dir)
