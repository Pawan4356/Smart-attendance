# attendance_log/log_attendance.py

from datetime import datetime
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["COD1"]
attendance_col = db["attendance"]

def log_attendance_bulk(student_ids, class_id):
    """
    Logs attendance for multiple student_ids for today's date in class `COD1`.
    """
    today = datetime.today().strftime("%Y-%m-%d")
    for student_id in student_ids:
        record = {
            "student_id": student_id,
            "class_id": class_id,
            "timestamp": datetime.now(),
            "date": today,
            "status": "Present"
        }

        # Avoid duplicate entries for same student on same day
        if not attendance_col.find_one({"student_id": student_id, "class_id": class_id, "date": today}):
            attendance_col.insert_one(record)
            print(f"Marked present: {student_id}")
        else:
            print(f"Already marked: {student_id}")
