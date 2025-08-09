from inference.car_detector import CarDetector

class CarPipeline:
    def __init__(self, model_path, device=None):
        self.detector = CarDetector(model_path=model_path, device=device)

    def run(self, image_path):
        results = self.detector.predict(image_path)
        return results



'''script

import os
import cv2


def crop_and_save_cars_from_folder(input_folder, output_folder, car_pipeline):
    os.makedirs(output_folder, exist_ok=True)

    # Przechodzimy przez pliki w folderze
    for filename in os.listdir(input_folder):
        if not (filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg')):
            continue

        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Nie można wczytać {image_path}")
            continue

        # Uruchamiamy detektor
        results = car_pipeline.run(image_path)

        # Zakładam, że results to lista wykryć z bbox w results["carbox"]
        car_detections = [det for det in results if det.get("label") == "car"]

        if not car_detections:
            print(f"Brak wykrytych samochodów w {filename}")
            continue

        # wybierz wykrycie z największym bbox
        largest_car = max(car_detections,
                          key=lambda det: (det["bbox"][2] - det["bbox"][0]) * (det["bbox"][3] - det["bbox"][1]))

        bbox = largest_car["bbox"]
        x1, y1, x2, y2 = map(int, bbox)
        cropped_car = image[y1:y2, x1:x2]


        # Tworzymy nazwę pliku wynikowego
        base_name, ext = os.path.splitext(filename)
        out_name = f"{base_name}{ext}"
        out_path = os.path.join(output_folder, out_name)

        cv2.imwrite(out_path, cropped_car)
        print(f"Zapisano: {out_path}")


if __name__ == "__main__":
    input_folder = "../../Datasets/data/dataset_views/train/side"
    output_folder = "../../Datasets/data/dataset_views/train/side1"

    # Tutaj powinieneś mieć swój obiekt car_pipeline (np. detektor)
    car_pipeline = CarPipeline('../config/car_detector.pt')

    # Przykład wywołania funkcji:
    crop_and_save_cars_from_folder(input_folder, output_folder, car_pipeline)'''