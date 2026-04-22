import cv2
import time
from ingestion.detect_motion import detect_motion

# Threshold for motion detection
THRESHOLD = 7
# Time to wait after motion is detected
CAPTURE_DELAY = 1.5
#Cooldown before detecting motion again
COOLDOWN = 3

def frame(cap) :
    last_motion_time = 0
    prev_frame = None
    motion_detected_time = None

    while True : 
        ret, frame = cap.read()

        if not ret :
            return None
        
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is None :
            prev_frame = grey
            continue

        motion = detect_motion(prev_frame, grey, THRESHOLD)

        now = time.time()

        if motion and motion_detected_time is None and ((now - last_motion_time) > COOLDOWN):
            motion_detected_time = now
            
        if motion_detected_time is not None and ((now - motion_detected_time) >= CAPTURE_DELAY):
            last_motion_time = now
            motion_detected_time = None
            yield frame
            
        prev_frame = grey