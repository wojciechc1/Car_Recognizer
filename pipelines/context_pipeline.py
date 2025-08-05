from inference.context_classifier import ContextClassifier
from utils.dataset import val_transform
from PIL import Image

class ContextPipeline:
    def __init__(self, model_path):
        self.classifier = ContextClassifier(model_path)

    def run(self, image_path):
        img = Image.open(image_path).convert("RGB")
        img_tensor = val_transform(img)
        label, probs = self.classifier.predict(img_tensor)
        return label, probs
