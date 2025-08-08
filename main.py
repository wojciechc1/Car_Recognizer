from pipelines.main_pipeline import CarAnalysisPipeline
from utils.save_result import save_result_to_json

pipeline = CarAnalysisPipeline(paths={
    "view": "./config/view_classifier.pth",
    "logo": "./scripts/runs/detect/train2/weights/best.pt",
    "context": "./config/context_classifier.pth",
    # "type": "...",
    # "model": "..."
})

results = pipeline.run("./test2.jpg")

save_result_to_json(results, "result.json")