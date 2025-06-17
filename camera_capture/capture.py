import cv2
import schedule
import time
from datetime import datetime

TIME = 15

def capture_frame():
    cam = cv2.VideoCapture("rtsp://your_wificam_url")
    ret, frame = cam.read()
    if ret:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        cv2.imwrite(f"captures/{timestamp}.jpg", frame)
    cam.release()

schedule.every(TIME).minutes.do(capture_frame)

while True:
    schedule.run_pending()
    time.sleep(1)
