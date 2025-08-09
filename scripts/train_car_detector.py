from inference.car_detector import CarDetector
import os

if __name__ == '__main__': # for multiprocessing
    detector = CarDetector()
    detector.train(data_yaml='../../Datasets/data/car_bbox/cars bbox.v1i.yolov8/data.yaml', epochs=10)

#results = detector.predict('..')
#results[0].show()
