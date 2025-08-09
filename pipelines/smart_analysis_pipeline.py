'''


1. ViewClassifier -> 'front', 'side', 'rear', 'angled'

- (front) - ContextClassifier - LogoDetector
- (side) - CarTypeClassifier
- (back) - [ContextClassifier] - LogoDetector - ModelDetector
- /future/'----(angled) - CarTypeClassifier - LogoDetector

'''
from inference import car_detector
from pipelines.view_pipeline import ViewPipeline
from pipelines.logo_detector_pipeline import LogoPipeline
from pipelines.context_pipeline import ContextPipeline
#from utils.calculate_score import calculate_score
from pipelines.car_detector_pipeline import CarPipeline
from pipelines.color_pipeline import ColorClassifierPipeline

class CarAnalysisPipeline:
    def __init__(self, paths):
        self.view_classifier = ViewPipeline(model_path=paths["view"])
        self.logo_detector = LogoPipeline(model_path=paths["logo"])
        self.context_classifier = ContextPipeline(model_path=paths["context"])
        self.car_detector = CarPipeline(model_path=paths["car"])
        self.color_detector = ColorClassifierPipeline(model_path=paths["color"])
        # self.car_type_classifier = CarTypeClassifierPipeline(model_path=paths["type"])
        # self.model_detector = ModelDetectorPipeline(model_path=paths["model"])

    def run(self, image_path):
        result = {}

        # 1. VIEW CLASSIFICATION
        view = self.view_classifier.run(image_path)
        result["view"] = view
        result.update({
            "view": view
        })
        carbox = self.car_detector.run(image_path)
        if not carbox:
            # brak wykrytych aut, obsłuż wyjątek lub zwróć None
            result.update({"carbox": [], "color": None})
        else:
            result.update({"carbox": carbox})
            color = self.color_detector.run(image_path, carbox)
            result.update({'color': color})

        # 2. BRANCHING BASED ON VIEW
        if view["label_name"] == "front":
            context_result = self.context_classifier.run(image_path)
            logo_result = self.logo_detector.run(image_path)
            result.update({
                "context": context_result,
                "logo": logo_result
            })

        elif view["label_name"] == "side":
            # type_result = self.car_type_classifier.run(image_path)
            logo_result = self.logo_detector.run(image_path)
            result.update({
                # "car_type": type_result
                "logo": logo_result

            })

        elif view["label_name"] == "rear":
            # context_result = self.context_classifier.run(image_path)
            logo_result = self.logo_detector.run(image_path)
            # model_result = self.model_detector.run(image_path)
            result.update({
                # "context": context_result,
                "logo": logo_result,
                # "model": model_result
            })

        # 1. Uruchamiasz car detector
        #car_result = car_detector_pipeline.run("images/test.jpg")
        # car_result np. = {"label": "car", "bbox": [50, 100, 300, 400], "score": 0.98}

        # 2. Przepuszczasz przez color classifier
        #color_result = color_classifier_pipeline.run("images/test.jpg", car_result["bbox"])
        #result.update({
        #    "color": color_result
        #})
        #result = calculate_score(result)
        return result