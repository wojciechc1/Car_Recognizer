from inference.context_classifier import ContextClassifier
from utils.dataset import val_transform
from PIL import Image


class ContextPipeline:
    def __init__(self, model_path, class_names=None, device=None):
        self.classifier = ContextClassifier(model_path, device)
        self.class_names = class_names or ["audi", "toyota", "bmw"]  # <- dopasuj do siebie

    def run(self, image_path):
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Error loading image: {e}")

        img_tensor = val_transform(img)
        label_idx, probs = self.classifier.predict(img_tensor)
        label_name = self.class_names[label_idx]

        return {
            "label_idx": label_idx,
            "label_name": label_name,
            "probabilities": probs.squeeze().tolist()
        }
