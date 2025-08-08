from inference.car_detector import CarDetector

class LogoPipeline:
    def __init__(self, model_path, device=None):
        self.detector = CarDetector(model_path=model_path, device=device)

    def run(self, image_path):
        results = self.detector.predict(image_path)
        return results
