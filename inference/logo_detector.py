import torch
from ultralytics import YOLO

class LogoDetector:
    def __init__(self, model_path=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_path:
            self.model = YOLO(model_path)  # Twój wytrenowany model
        else:
            self.model = YOLO('yolov8n.pt')  # Model pretrenowany
        self.model.to(self.device)

    def train(self, data_yaml, epochs=10, imgsz=640):
        """
        Wrapper do treningu modelu YOLOv8 z ultralytics.
        data_yaml: ścieżka do pliku z danymi (train/test/classes)
        """
        self.model.train(data=data_yaml, epochs=epochs, imgsz=imgsz)

    def predict(self, image):
        # image może być ścieżką do pliku lub np. numpy array (OpenCV image)
        results = self.model(image)
        return results