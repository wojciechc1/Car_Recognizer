import torch
from ultralytics import YOLO
import cv2

class CarDetector:
    def __init__(self, model_path=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_path:
            self.model = YOLO(model_path)  # Twój wytrenowany YOLO (np. tylko do aut)
        else:
            self.model = YOLO('yolov8n.pt')  # Pretrenowany YOLO (wszystkie klasy COCO)
        self.model.to(self.device)


    def train(self, data_yaml, epochs=10, imgsz=640):
        """
        Wrapper do treningu modelu YOLOv8 z ultralytics.
        data_yaml: ścieżka do pliku z danymi (train/test/classes)
        """
        self.model.train(data=data_yaml, epochs=epochs, imgsz=imgsz)


    def predict(self, image):
        results = self.model(image)
        detections = []
        for box in results[0].boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls_id = box
            label = results[0].names[int(cls_id)]
            detections.append({
                "label": label,
                "confidence": float(conf),
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })
        return detections
