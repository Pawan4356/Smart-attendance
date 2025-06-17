import os
import cv2
import torch

import numpy as np
import pandas as pd
import requests
import datetime
from ultralytics import YOLO
import onnxruntime as ort

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Download the files using requests
def download_file(url, filename):
    if os.path.exists(filename):
        print(f"{filename} already exists, skipping download")
        return
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded {filename}")

# Download YOLOv8 face model
    # download_file(
    #     'https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8n-face-lindevs.pt',
    #     'models/yolov8n-face.pt'
    # )

    # # Download ArcFace facial analysis model
    # download_file(
    #     'https://github.com/yakhyo/facial-analysis/releases/download/v0.0.1/w600k_r50.onnx',
    #     'models/w600k_r50.onnx'
    # )

# Load YOLOv8 face detection model
yolo_model = YOLO("models/yolov11n-face.pt")
print("YOLO is using CUDA:", torch.cuda.is_available())


# Load ArcFace (ResNet-50) model for face embeddings
arcface_session = ort.InferenceSession("models/w600k_r50.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
input_name = arcface_session.get_inputs()[0].name
print("ArcFace is using:", arcface_session.get_providers())


# Create directory for known faces
os.makedirs("known_faces", exist_ok=True)
print("Put subfolders of face images inside 'known_faces/'.")

# Process known faces and create embeddings
known_embeddings = {}
for person_name in os.listdir("known_faces"):
    person_dir = os.path.join("known_faces", person_name)
    if not os.path.isdir(person_dir):
        continue
    embeddings = []
    for img_name in os.listdir(person_dir):
        if img_name.startswith('.'):
            continue
        img_path = os.path.join(person_dir, img_name)
        if os.path.isdir(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        results = yolo_model(img)
        if len(results) == 0 or len(results[0].boxes) == 0:
            print(f"No face detected in {img_path}, skipping.")
            continue
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        best_idx = int(np.argmax(confs))
        x1, y1, x2, y2 = boxes[best_idx].astype(int)
        face_img = img[y1:y2, x1:x2]
        if face_img.size == 0:
            continue
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (112, 112))
        face_norm = (face_resized - 127.5) * 0.0078125
        face_input = np.transpose(face_norm, (2, 0, 1))
        face_input = face_input[None, :, :, :].astype(np.float32)
        embedding = arcface_session.run(None, {input_name: face_input})[0].flatten()
        embeddings.append(embedding)
    if embeddings:
        mean_emb = np.mean(embeddings, axis=0)
        mean_emb /= np.linalg.norm(mean_emb)
        known_embeddings[person_name] = mean_emb
        print(f"Built embedding for {person_name} using {len(embeddings)} images.")

# Initialize attendance file with header if not exists
attendance_file = "attendance.csv"
if not os.path.exists(attendance_file):
    with open(attendance_file, 'w') as f:
        f.write("Name,Timestamp\n")

# Start webcam for real-time recognition
cap = cv2.VideoCapture(0)
recognized = set()

print("Starting webcam. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame)

    if len(results) > 0 and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        for i, (box, conf) in enumerate(zip(boxes, confs)):
            if conf < 0.5:
                continue

            x1, y1, x2, y2 = box.astype(int)
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue

            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (112, 112))
            face_norm = (face_resized - 127.5) * 0.0078125
            face_input = np.transpose(face_norm, (2, 0, 1))
            face_input = face_input[None, :, :, :].astype(np.float32)

            embedding = arcface_session.run(None, {input_name: face_input})[0].flatten()
            embedding /= np.linalg.norm(embedding)

            best_match = None
            best_score = -1
            for name, known_emb in known_embeddings.items():
                similarity = np.dot(embedding, known_emb)
                if similarity > best_score:
                    best_score = similarity
                    best_match = name

            label = "Unknown"
            color = (0, 0, 255)

            if best_score > 0.35:
                label = f"{best_match} ({best_score:.2f})"
                color = (0, 255, 0)
                today = datetime.datetime.now().strftime("%Y-%m-%d")
                person_today = f"{best_match}_{today}"
                if person_today not in recognized:
                    with open(attendance_file, 'a') as f:
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        f.write(f"{best_match},{timestamp}\n")
                    recognized.add(person_today)

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show video output
    cv2.imshow("Face Recognition Attendance", frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
