from inference.logo_detector import LogoDetector


detector = LogoDetector()
detector.train(data_yaml='../data/dataset_logo_bb/yolo_ds/data.yaml', epochs=50)

results = detector.predict('../data/dataset/train/audi/front/image_0.jpg')
results[0].show()
