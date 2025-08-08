from pipelines.main_pipeline import CarAnalysisPipeline


pipeline = CarAnalysisPipeline(paths={
    "view": "./config/view_classifier.pth",
    "logo": "./scripts/runs/detect/train2/weights/best.pt",
    # "context": "...",
    # "type": "...",
    # "model": "..."
})

results = pipeline.run("./audi_test.jpeg")
print(results)