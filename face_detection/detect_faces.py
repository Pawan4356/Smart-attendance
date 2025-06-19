import os
import cv2
import numpy as np
from ultralytics import YOLO
import onnxruntime as ort
from datetime import datetime

# --- Load Models ---
yolo_model = YOLO("models/yolov8n-face.pt")
    
# --- Constants ---
INPUT_DIR = "captures"
FACE_SIZE = (112, 112)

def preprocess_face(face_img):
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, FACE_SIZE)
    face_norm = (face_resized - 127.5) * 0.0078125
    face_input = np.transpose(face_norm, (2, 0, 1))[None].astype(np.float32)
    return face_input

def detect_and_process(class_id, threshold):
    if not os.listdir(INPUT_DIR):
        print("No images found in captures/")
        return

    latest_img = sorted(os.listdir(INPUT_DIR))[-1]
    img_path = os.path.join(INPUT_DIR, latest_img)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read image: {img_path}")
        return

    results = yolo_model(img)
    boxes = results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes) > 0 else []
    confs = results[0].boxes.conf.cpu().numpy() if len(results[0].boxes) > 0 else []

    faces = []
    for i, (box, conf) in enumerate(zip(boxes, confs)):
        if conf < 0.5:
            continue
        x1, y1, x2, y2 = map(int, box)
        face_img = img[y1:y2, x1:x2]
        if face_img.size == 0:
            continue
        processed_face = preprocess_face(face_img)
        faces.append(processed_face)

    if not faces:
        print("No valid faces detected.")
        return

    return faces

    # for i, (box, student_id) in enumerate(zip(boxes, recognized_ids)):
    #     x1, y1, x2, y2 = map(int, box)
    #     label = student_id if student_id else "Unknown"
    #     color = (0, 255, 0) if student_id else (0, 0, 255)
    #     cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    #     cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # # Save the annotated image
    # save_path = f"annotated/{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    # os.makedirs("annotated", exist_ok=True)
    # cv2.imwrite(save_path, img)
    # print(f"Saved annotated result to {save_path}")
