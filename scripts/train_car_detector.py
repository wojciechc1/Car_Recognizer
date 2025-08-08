from inference.car_detector import CarDetector


detector = CarDetector()
detector.train(data_yaml='..', epochs=50)

results = detector.predict('..')
results[0].show()
