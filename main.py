from ingestion.video_feed import frame
import cv2


cap = cv2.VideoCapture(0)

for frames in frame(cap):
    #print("Motion Detected")
    cv2.imshow("Frame", frames)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break