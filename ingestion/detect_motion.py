import cv2

def detect_motion(prev_frame, grey, THRESHOLD):

    diff = cv2.absdiff(prev_frame, grey)
    motion = diff.mean()
    return motion > THRESHOLD