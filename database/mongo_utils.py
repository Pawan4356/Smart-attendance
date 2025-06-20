# database/mongo_utils.py

import numpy as np
from datetime import datetime
from pymongo import MongoClient

MONGO_URI = "mongodb://localhost:27017"
client = MongoClient(MONGO_URI)

MAX_EMBEDDINGS = 6

# Insert or update a student (dynamic DB based on class_id)
def insert_student(student_id, name, class_id, embedding):
    if isinstance(new_embeddings, np.ndarray):
        new_embeddings = [new_embeddings.tolist()]
    elif isinstance(new_embeddings, list) and isinstance(new_embeddings[0], np.ndarray):
        new_embeddings = [e.tolist() for e in new_embeddings]

    # Use class_id as the database name
    db = client[class_id]
    students_col = db["students"]

    existing = students_col.find_one({"student_id": student_id})

    if existing and "embeddings" in existing:
        current_embs = existing["embeddings"]
        if len(current_embs) >= MAX_EMBEDDINGS:
            print(f"Student {student_id} already has {len(current_embs)} embeddings. Skipping.")
            return

        remaining = MAX_EMBEDDINGS - len(current_embs)
        new_to_add = new_embeddings[:remaining]
        updated_embs = current_embs + new_to_add

        students_col.update_one(
            {"student_id": student_id},
            {"$set": {
                "name": name,
                "class_id": class_id,
                "embeddings": updated_embs,
                "type": "student"
            }},
            upsert=True
        )
        print(f"Updated {student_id}: added {len(new_to_add)} embeddings (total: {len(updated_embs)}).")

    else:
        students_col.update_one(
            {"student_id": student_id},
            {"$set": {
                "name": name,
                "class_id": class_id,
                "embedding": embedding,
                "registered_at": datetime.utcnow(),
                "type": "student"
            }},
            upsert=True
        )
        print(f"Inserted new student {student_id} with {len(new_embeddings)} embeddings.")

def get_student_by_id(class_id, student_id):
    db = client[class_id]
    col = db["students"]
    return col.find_one({"student_id": student_id})
