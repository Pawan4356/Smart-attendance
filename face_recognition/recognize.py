import faiss
import numpy as np
from database.mongo_utils import get_all_embeddings

def build_faiss_index():
    """
    Loads all embeddings from MongoDB and builds a FAISS index.
    Returns the FAISS index and corresponding student_id labels.
    """
    records = get_all_embeddings()
    if not records:
        raise ValueError("No embeddings found in database.")

    embeddings = np.array([r["embedding"] for r in records], dtype=np.float32)
    labels = [r["student_id"] for r in records]

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index, labels

def recognize_faces(face_embeddings, index, labels, threshold=0.35):
    """
    Recognizes multiple faces and returns a list of matched student_ids.
    """
    face_embeddings = np.array(face_embeddings, dtype=np.float32)
    D, I = index.search(face_embeddings, 1)

    matched_ids = []
    for d, idx in zip(D, I):
        if d[0] >= threshold:
            matched_ids.append(labels[idx[0]])
    return matched_ids
