from ultralytics import YOLO
import cv2
import numpy as np
from paddleocr import PaddleOCRVL

class PasstoYOLO :
    def __init__ (self, yolo_path) :
        self.model = YOLO(yolo_path)

    def license_coordinates (self, frame) :
        results = self.model([frame], stream=True)
        for result in results :
            boxes = result.boxes
            x1, y1, x2, y2 = boxes.xyxy.numpy()
            return x1,y1,x2,y2

    def crop_into_plate (self, frame, x1, y1, x2, y2) :
        plate_img = frame[int(y1):int(y2), int(x1):int(x2)]
        return plate_img
    
class PaddleInference :

    def ocr_inference (self, plate_img) :
        pipeline = PaddleOCRVL(use_angle_cls=True, lang='en')
        output = pipeline.predict(plate_img)
        for res in output :
            print(res)
            return res
