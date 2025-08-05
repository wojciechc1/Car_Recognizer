from inference.logo_detector import LogoDetector

detector = LogoDetector()
detector.train(data_yaml='path/to/data.yaml', epochs=20)
model.train(data='ścieżka/do/data.yaml', epochs=10, imgsz=640)

results = detector.predict('test_image.jpg')
print(results)

