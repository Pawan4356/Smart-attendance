# import schedule
# import time
# from camera_capture.capture import capture_frame
# from face_detection.detect_faces import detect_and_process


# def scheduled_task():
#     capture_frame()
#     detect_and_process()

# # Schedule to run every 15 minutes
# schedule.every(15).minutes.do(scheduled_task)

# if __name__ == "__main__":
#     print("Face recognition attendance system started. Press Ctrl+C to stop.")
#     while True:
#         schedule.run_pending()
#         time.sleep(1)

# ===== Temporary for testing with an image 👇 ======

import time
import cv2
from training.fine_tune import start_scheduler
from face_detection.detect_faces import detect_and_process
from face_recognition.arcface_model import get_arcface_embedding
from face_recognition.recognize import build_faiss_index, recognize_faces
from attendance_log.log_attendance import log_attendance_bulk

def test_with_static_image(image_path="faces/Pawan.jpg"):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Image not found: {image_path}")
        return

    faces = detect_and_process()
    if not faces:
        print("No faces detected.")
        return

    embeddings = [get_arcface_embedding(face) for face in faces]
    index, labels = build_faiss_index()
    recognized_ids = recognize_faces(embeddings, index, labels)

    if recognized_ids:
        print(f"Recognized IDs: {recognized_ids}")
        log_attendance_bulk(recognized_ids)
    else:
        print("No known faces recognized.")

if __name__ == "__main__":
    start_scheduler()
    test_with_static_image()

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Stopping scheduler and exiting.")
