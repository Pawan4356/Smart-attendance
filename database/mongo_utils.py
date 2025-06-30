import numpy as np
from datetime import datetime
from pymongo import MongoClient


class MongoHandler:
    def __init__(self, uri, db_name):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.students = self.db["students"]
        self.attendance = self.db["attendance"]

    def find_student_by_name(self, name):
        return self.students.find_one({"name": name})
    
    def get_student_by_id(self, class_id, student_id):
        col = self.db["students"]
        return col.find_one({"student_id": student_id})

    def mark_attendance(self, student_id, class_id):
        record = {
            "student_id": student_id,
            "timestamp": datetime.utcnow(),
            "class_id": class_id
        }
        self.attendance.insert_one(record)
        print(f"[DB] Attendance logged for {student_id} at {record['timestamp']}")

    def insert_student(self, student_id, name, class_id, new_embeddings):
        # Normalize embeddings to list of lists
        if isinstance(new_embeddings, np.ndarray):
            new_embeddings = [new_embeddings.tolist()]
        elif isinstance(new_embeddings, list):
            new_embeddings = [e.tolist() if isinstance(e, np.ndarray) else e for e in new_embeddings]

        # Select class-specific database and collection
        student_col = self.client[class_id]["students"]

        student = student_col.find_one({"student_id": student_id})

        if student and "embeddings" in student:
            current_embeddings = student["embeddings"]
            if len(current_embeddings) >= 6:
                print(f"Student {student_id} already has {len(current_embeddings)} embeddings. Skipping.")
                return

            new_to_add = new_embeddings[:6 - len(current_embeddings)]
            updated_embeddings = current_embeddings + new_to_add

            student_col.update_one(
                {"student_id": student_id},
                {"$set": {
                    "name": name,
                    "class_id": class_id,
                    "embeddings": updated_embeddings,
                    "type": "student"
                }}
            )
            print(f"Updated {student_id}: added {len(new_to_add)} embeddings (total: {len(updated_embeddings)}).")
        else:
            student_col.update_one(
                {"student_id": student_id},
                {"$set": {
                    "name": name,
                    "class_id": class_id,
                    "embeddings": new_embeddings,
                    "registered_at": datetime.utcnow(),
                    "type": "student"
                }},
                upsert=True
            )
            print(f"Inserted new student {student_id} with {len(new_embeddings)} embeddings.")

    def get_all_registered_students(self):
        return list(self.db["students"].find({"type": "student"}))

