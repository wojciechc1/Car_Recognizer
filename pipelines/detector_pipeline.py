from inference.logo_detector import LogoDetector


detector = LogoDetector(model_path='../scripts/runs/detect/train2/weights/best.pt')

results = detector.predict('../audi_test.jpeg')

print(results)