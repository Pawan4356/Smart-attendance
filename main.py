import time
import schedule 
from camera_capture.capture import capture_frame
from face_detection.detect_faces import detect_and_process
from face_recognition.recognize import recognize_faces_mongo
from attendance_log.log_attendance import log_attendance_bulk
from database.enroll_student import enroll_students_from_folder
from face_recognition.arcface_model import get_arcface_embedding

class_id = "COD1"
threshold = 0.35

def scheduled_task():
    capture_frame()
    faces = detect_and_process(class_id, threshold)
    if not faces:
        print("No faces detected.")
        return
    embeddings = [get_arcface_embedding(face) for face in faces]
    recognized_ids = recognize_faces_mongo(embeddings, class_id, threshold)
    log_attendance_bulk(recognized_ids, class_id)

# if __name__ == "__main__":
#     print("Enrolling known students from folder...")
#     enroll_students_from_folder()

#     print("Face recognition attendance system started. Press Ctrl+C to stop.")
#     schedule.every(15).minutes.do(scheduled_task)

#     while True:
#         schedule.run_pending()
#         time.sleep(1)

# ===== To run a single time =====
if __name__ == "__main__":
    print("Enrolling known students from folder...")
    enroll_students_from_folder()

    print("Running face recognition attendance once...")
    scheduled_task()
