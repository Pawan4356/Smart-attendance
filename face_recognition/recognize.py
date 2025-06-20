import numpy as np
from pymongo import MongoClient

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def recognize_faces_mongo(face_embeddings, class_id, threshold=0.35):
    """
    Recognizes faces by comparing embeddings against all stored embeddings in MongoDB.

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

    records = list(students_col.find({}, {"_id": 0, "student_id": 1, "embeddings": 1}))
    if not records:
        raise ValueError(f"No embeddings found in database '{class_id}'.")

    matched_ids = []

    for test_emb in face_embeddings:
        best_match_id = None
        best_score = -1

        for record in records:
            student_id = record["student_id"]
            student_embeddings = record.get("embeddings", [])

            for emb in student_embeddings:
                score = cosine_similarity(test_emb, np.array(emb, dtype=np.float32))
                if score > best_score:
                    best_score = score
                    best_match_id = student_id

        if best_score >= threshold:
            matched_ids.append(best_match_id)
        else:
            matched_ids.append(None)

    return matched_ids

