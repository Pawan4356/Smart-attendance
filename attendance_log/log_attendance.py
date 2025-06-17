# attendance_log/log_attendance.py

from datetime import datetime
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")

def log_attendance_bulk(student_ids, class_id):
    """
    Logs attendance for multiple student_ids in the database named after `class_id`.
    Collection name remains 'attendance'.
    Avoids duplicate entries for the same student on the same date.
    """
    db = client[class_id]  # Use class_id as database name
    attendance_col = db["attendance"]
    today = datetime.today().strftime("%Y-%m-%d")

    for student_id in student_ids:
        record = {
            "student_id": student_id,
            "class_id": class_id,
            "timestamp": datetime.now(),
            "date": today,
            "status": "Present"
        }

        if not attendance_col.find_one({"student_id": student_id, "class_id": class_id, "date": today}):
            attendance_col.insert_one(record)
            print(f"Marked present: {student_id}")
        else:
            print(f"Already marked: {student_id}")
