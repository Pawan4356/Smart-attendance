import os
import cv2
import time
import schedule
from datetime import datetime

TIME = 15

# def capture_frame():
#     cam = cv2.VideoCapture("rtsp://your_wificam_url")
#     ret, frame = cam.read()
#     if ret:
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M")
#         cv2.imwrite(f"captures/{timestamp}.jpg", frame)
#     cam.release()

# Schedule the capture
# schedule.every(TIME).minutes.do(capture_frame)
# print(f"Running scheduled webcam capture every {TIME} minutes...")
# while True:
#     schedule.run_pending()
#     time.sleep(1)

# ===== Temporary function for testing =====
os.makedirs("captures", exist_ok=True)

def capture_frame():
    # Use laptop's built-in camera (device 0)
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Failed to open laptop webcam.")
        return

    ret, frame = cam.read()
    if ret:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"captures/{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Captured frame saved to {filename}")
    else:
        print("Failed to capture frame from webcam.")
    
    cam.release()