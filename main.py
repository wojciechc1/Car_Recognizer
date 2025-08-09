from pipelines.smart_analysis_pipeline import CarAnalysisPipeline
from pipelines.complex_analysis_pipeline import ComplexCarAnalysisPipeline
from utils.save_result import save_result_to_json

smart_pipeline = CarAnalysisPipeline(paths={
    "view": "./config/view_classifier.pth",
    "logo": "./scripts/runs/detect/train2/weights/best.pt",
    "context": "./config/context_classifier.pth",
    "car": "./config/car_detector.pt",
    "color": "./config/color_classifier.pth"
    # "type": "...",
    # "model": "..."
})

results = smart_pipeline.run("./test2.jpg")

save_result_to_json(results, "result.json")


complex_pipeline = ComplexCarAnalysisPipeline(paths={
    "view": "./config/view_classifier.pth",
    "logo": "./scripts/runs/detect/train2/weights/best.pt",
    "context": "./config/context_classifier.pth",
    "car": "./config/car_detector.pt",
    "color": "./config/color_classifier.pth"
    # "type": "...",
    # "model": "..."
})


# Po zapisaniu 3 zdjęć do tymczasowych plików:
temp_paths = {"front": 'testf.jpg', "side": 'tests.jpg', "rear": 'testb.jpg'}
results = complex_pipeline.run(temp_paths)
save_result_to_json(results, "result.json")
#st.json(results)