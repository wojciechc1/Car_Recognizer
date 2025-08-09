from pipelines.view_pipeline import ViewPipeline
from pipelines.logo_detector_pipeline import LogoPipeline
from pipelines.context_pipeline import ContextPipeline
from utils.combine_results import combine_results
from pipelines.car_detector_pipeline import CarPipeline
from pipelines.color_pipeline import ColorClassifierPipeline



# Opcjonalnie możesz rozdzielić to do osobnego pliku np. pipelines/complex_pipeline.py
class ComplexCarAnalysisPipeline:
    def __init__(self, paths):
        self.view_classifier = ViewPipeline(model_path=paths["view"])
        self.logo_detector = LogoPipeline(model_path=paths["logo"])
        self.context_classifier = ContextPipeline(model_path=paths["context"])
        self.car_detector = CarPipeline(model_path=paths["car"])
        self.color_detector = ColorClassifierPipeline(model_path=paths["color"])
        # self.car_type_classifier = CarTypeClassifierPipeline(model_path=paths["type"])
        # self.model_detector = ModelDetectorPipeline(model_path=paths["model"])

    def run(self, images_dict):
        """
        images_dict: dict o kluczach "front", "side", "rear" i ścieżkach do plików
        """
        result = {}

        # FRONT IMAGE ANALYSIS
        if "front" in images_dict:
            front_path = images_dict["front"]
            view = self.view_classifier.run(front_path)
            context = self.context_classifier.run(front_path)
            logo = self.logo_detector.run(front_path)
            carbox = self.car_detector.run(front_path)
            color = self.color_detector.run(front_path, carbox)
            result["front"] = {
                "view": view,
                "context": context,
                "logo": logo,
                "carbox": carbox,
                "color": color
            }

        # SIDE IMAGE ANALYSIS
        if "side" in images_dict:
            side_path = images_dict["side"]
            view = self.view_classifier.run(side_path)
            # car_type = self.car_type_classifier.run(side_path)
            logo = self.logo_detector.run(side_path)
            carbox = self.car_detector.run(side_path)
            color = self.color_detector.run(side_path, carbox)
            result["side"] = {
                "view": view,
                # "car_type": car_type,
                "logo": logo,
                "carbox": carbox,
                "color": color
            }

        # REAR IMAGE ANALYSIS
        if "rear" in images_dict:
            rear_path = images_dict["rear"]
            view = self.view_classifier.run(rear_path)
            # context = self.context_classifier.run(rear_path)
            logo = self.logo_detector.run(rear_path)
            # model = self.model_detector.run(rear_path)
            carbox = self.car_detector.run(rear_path)
            color = self.color_detector.run(rear_path, carbox)

            result["rear"] = {
                "view": view,
                # "context": context,
                "logo": logo,
                # "model": model,
                "carbox": carbox,
                "color": color
            }

        #result = combine_results(result)
        return result
