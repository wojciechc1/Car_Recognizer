from inference.color_classifier import ColorClassifier
from utils.dataset import val_transform
from PIL import Image


class ColorClassifierPipeline:
    def __init__(self, model_path, class_names=None, device=None):
        self.classifier = ColorClassifier(model_path, device)

        self.class_names = class_names or ['black', 'blue', 'gray', 'other', 'red', 'white', 'yellow']

    def run(self, image_path, carbox):
        """
        image_path: ścieżka do pliku z obrazem
        bbox: [x_min, y_min, x_max, y_max] w pikselach
        """

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Error loading image: {e}")

        largest_car = max(carbox,
                          key=lambda det: (det["bbox"][2] - det["bbox"][0]) * (det["bbox"][3] - det["bbox"][1]))

        bbox = largest_car["bbox"]

        # Wycinanie bounding boxa auta
        x_min, y_min, x_max, y_max = map(int, bbox)
        cropped_img = img.crop((x_min, y_min, x_max, y_max))

        img_tensor = val_transform(cropped_img)
        label_idx, probs = self.classifier.predict(img_tensor)
        label_name = self.class_names[label_idx]

        return {
            "label_idx": label_idx,
            "label_name": label_name,
            "probabilities": probs.squeeze().tolist()
        }
