import os
import cv2
import time
import schedule
import datetime
from ultralytics.utils import LOGGER
from database.mongo_utils import MongoHandler
from face_detection.detector import FaceDetector
from initialEnrollment.enroll import enroll_students
from face_recognition.recognizer import FaceRecognizer

LOGGER.setLevel("ERROR")
class_id = "COD1"
threshold = 0.1
uri = "mongodb://localhost:27017"
face_detector = FaceDetector("models/yolov11n-face.pt")
mongo_handler = MongoHandler(uri, class_id)
face_recognizer = FaceRecognizer("models/w600k_r50.onnx", mongo_handler, threshold)

def schedule():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Webcam not accessible")
        exit()

    recognized_today = set()
    today = datetime.datetime.now().strftime("%Y-%m-%d")

    print("Starting camera. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Frame not captured.")
            break

        boxes, confs = face_detector.detect_faces(frame)
        for (x1, y1, x2, y2), conf in zip(boxes, confs):
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue

            result = face_recognizer.recognize(face_img)
            if result:
                student_id, class_id, full_name, score = result
                nameList = full_name.split("_", 2)
                name = nameList[len(nameList) - 1].replace("_", " ")
                label = f"{name} ({score:.2f})"
                color = (0, 255, 0)

                key = f"{student_id}_{today}"
                if key not in recognized_today:
                    mongo_handler.mark_attendance(student_id, class_id)
                    recognized_today.add(key)
            else:
                label = "Unknown"
                color = (0, 0, 255)


            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Face Recognition Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

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
    enroll_students(class_id, threshold, uri)

    print("Running face recognition attendance once...")
    schedule()