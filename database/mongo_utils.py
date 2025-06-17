# # database/mongo_utils.py

# import numpy as np
# from pymongo import MongoClient
# from datetime import datetime

# # Replace with your actual Mongo URI
# MONGO_URI = "mongodb://localhost:27017"
# DB_NAME = "COD1"

# client = MongoClient(MONGO_URI)
# db = client[DB_NAME]
# students_col = db["students"]
# attendance_col = db["attendance"]

# # Insert or update a student
# def insert_student(student_id, name, class_id, embedding):
#     if isinstance(embedding, np.ndarray):
#         embedding = embedding.tolist()
#     students_col.update_one(
#         {"student_id": student_id},
#         {"$set": {
#             "name": name,
#             "class_id": class_id,
#             "embedding": embedding,
#             "registered_at": datetime.utcnow(),
#             "type": "student"
#         }},
#         upsert=True
#     )

# # Get all embeddings
# def get_all_embeddings():
#     return list(students_col.find({}, {"_id": 0, "student_id": 1, "embedding": 1}))

# # Log attendance
# def log_attendance(student_id, image_id):
#     now = datetime.utcnow()
#     attendance_col.insert_one({
#         "student_id": student_id,
#         "image_id": image_id,
#         "timestamp": now
#     })

# # Get recent attendance
# def get_attendance_records(student_id):
#     return list(attendance_col.find({"student_id": student_id}).sort("timestamp", -1))

# database/mongo_utils.py

import numpy as np
from pymongo import MongoClient
from datetime import datetime

MONGO_URI = "mongodb://localhost:27017"

client = MongoClient(MONGO_URI)

# Insert or update a student (dynamic DB based on class_id)
def insert_student(student_id, name, class_id, embedding):
    if isinstance(embedding, np.ndarray):
        embedding = embedding.tolist()

    # Use class_id as the database name
    db = client[class_id]
    students_col = db["students"]

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

# Get all embeddings from a specific class
def get_all_embeddings(class_id):
    db = client[class_id]
    students_col = db["students"]
    return list(students_col.find({}, {"_id": 0, "student_id": 1, "embedding": 1}))

# Log attendance (same DB used for logging as per class_id)
def log_attendance(student_id, class_id, image_id):
    db = client[class_id]
    attendance_col = db["attendance"]
    now = datetime.utcnow()
    attendance_col.insert_one({
        "student_id": student_id,
        "image_id": image_id,
        "timestamp": now
    })

# Get recent attendance for a student in a class
def get_attendance_records(student_id, class_id):
    db = client[class_id]
    attendance_col = db["attendance"]
    return list(attendance_col.find({"student_id": student_id}).sort("timestamp", -1))