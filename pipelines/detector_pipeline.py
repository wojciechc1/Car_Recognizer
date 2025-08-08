from inference.logo_detector import LogoDetector

class LogoPipeline:
    def __init__(self, model_path, device=None):
        self.detector = LogoDetector(model_path=model_path, device=device)

    def run(self, image_path):
        results = self.detector.predict(image_path)
        return results
