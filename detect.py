import cv2
from ultralytics import YOLO
import numpy as np

# Load YOLOv8/YOLOv11n face detection model
model = YOLO("models/yolov11n-face.pt")  # Update path as needed

def detect_faces(image_path):
    """
    Detect faces in an image and return cropped face images.
    """
    img = cv2.imread(image_path)
    results = model(img)

    faces = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # xyxy format
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = img[y1:y2, x1:x2]
            if face.size == 0:
                continue
            # Preprocess face: resize to 112x112 (ArcFace input size)
            face = cv2.resize(face, (112, 112))
            faces.append(face)

    return faces

if __name__ == "__main__":
    faces = detect_faces("classroom.jpg")
    print(f"Detected {len(faces)} faces")