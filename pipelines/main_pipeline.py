'''


1. ViewClassifier -> 'front', 'side', 'rear', 'angled'

- (front) - ContextClassifier - LogoDetector
- (side) - CarTypeClassifier
- (back) - [ContextClassifier] - LogoDetector - ModelDetector
- ////////////////'----(angled) - CarTypeClassifier - LogoDetector

'''

from pipelines.view_pipeline import ViewPipeline
from pipelines.detector_pipeline import LogoPipeline


class CarAnalysisPipeline:
    def __init__(self, paths):
        self.view_classifier = ViewPipeline(model_path=paths["view"])
        self.logo_detector = LogoPipeline(model_path=paths["logo"])
        # self.context_classifier = ContextClassifierPipeline(model_path=paths["context"])
        # self.car_type_classifier = CarTypeClassifierPipeline(model_path=paths["type"])
        # self.model_detector = ModelDetectorPipeline(model_path=paths["model"])

    def run(self, image_path):
        result = {}

        # 1. VIEW CLASSIFICATION
        view = self.view_classifier.run(image_path)
        result["view"] = view["label_name"]
        result["view_probs"] = view["probabilities"]

        # 2. BRANCHING BASED ON VIEW
        if view["label_name"] == "front":
            # context_result = self.context_classifier.run(image_path)
            logo_result = self.logo_detector.run(image_path)
            result.update({
                # "context": context_result,
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


        return result