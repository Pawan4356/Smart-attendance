import numpy as np
from pymongo import MongoClient

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def recognize_faces_mongo(face_embeddings, class_id, threshold=0.35):
    """
    Recognizes faces by comparing embeddings against those in MongoDB for a given class_id.
    Args:
        face_embeddings: List of face embeddings to recognize.
        class_id: MongoDB database name (e.g., "COD1").
        threshold: Cosine similarity threshold for a match.
    Returns:
        List of matched student_ids or None for each embedding.
    """
    client = MongoClient("mongodb://localhost:27017/")
    db = client[class_id]
    students_col = db["students"]

    # Fetch all student embeddings from the class-specific DB
    records = list(students_col.find({}, {"_id": 0, "student_id": 1, "embedding": 1}))
    if not records:
        raise ValueError(f"No embeddings found in database '{class_id}'.")

    student_ids = [r["student_id"] for r in records]
    stored_embeddings = np.array([r["embedding"] for r in records], dtype=np.float32)

    matched_ids = []

    for emb in face_embeddings:
        similarities = [cosine_similarity(emb, db_emb) for db_emb in stored_embeddings]
        max_index = np.argmax(similarities)
        max_score = similarities[max_index]

        if max_score >= threshold:
            matched_ids.append(student_ids[max_index])
        else:
            matched_ids.append(None)  # No match

    return matched_ids

# Temp Funcition

import cv2
from face_recognition.arcface_model import get_arcface_embedding

def capture_and_get_embedding():
    """
    Captures a single image from the laptop webcam, extracts and returns ArcFace embedding.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Failed to open webcam.")

    print("Press 's' to capture the image...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            continue

        cv2.imshow("Capture Face", frame)
        key = cv2.waitKey(1)
        if key == ord('s'):
            embedding = get_arcface_embedding(frame)
            break
        elif key == ord('q'):
            embedding = None
            break

    cap.release()
    cv2.destroyAllWindows()
    return embedding
