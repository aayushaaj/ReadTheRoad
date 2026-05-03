import os
from ingestion.video_feed import frame
from ocr.licensePlate import LicensePlateDetection, PaddleInference
from ocr.vehicleDetection import VehicleDetection
from ultralytics import YOLO
import cv2

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'models/license_plate.pt')

cap = cv2.VideoCapture(os.path.join(script_dir,'test_video1.mp4'))
LPD = LicensePlateDetection(model_path)
PI = PaddleInference()

for frames in frame(cap):
    coords = LPD.license_coordinates(frames)
    if coords is None:
        continue  # Skip this frame if no plate detected

    x1, y1, x2, y2 = coords
    print(f"Plate found at: {x1:.0f},{y1:.0f} -> {x2:.0f},{y2:.0f}")
    plate_img = LPD.crop_into_plate(frames, x1, y1, x2, y2)
    res = PI.ocr_inference(plate_img)
    print("OCR result: \n",res)
