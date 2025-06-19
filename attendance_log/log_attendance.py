# attendance_log/log_attendance.py

from datetime import datetime, time
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")

def get_current_period():
    """
    Returns the current period number based on time.
    Returns None during break or recess.
    """
    now = datetime.now().time()

    periods = [
        (time(9, 0), time(10, 0), 1),
        (time(10, 0), time(11, 0), 2),
        (time(11, 15), time(12, 15), 3),
        (time(12, 15), time(13, 15), 4),
        (time(14, 0), time(15, 0), 5),
        (time(15, 0), time(16, 0), 6),
        (time(17, 0), time(18, 0), 7), # Delete later
        (time(18, 0), time(19, 0), 8), # Delete later
    ]

    for start, end, period in periods:
        if start <= now < end:
            return period
    return None  # During break/recess or outside class hours

def log_attendance_bulk(student_ids, class_id):
    """
    Logs attendance for multiple student_ids based on current period.
    Skips logging if outside period time or during breaks.
    """
    current_period = get_current_period()
    if current_period is None:
        print("[⚠] No active period right now. Attendance not logged.")
        return

    db = client[class_id]
    attendance_col = db["attendance"]
    today = datetime.today().strftime("%Y-%m-%d")

    for student_id in student_ids:
        record = {
            "student_id": student_id,
            "class_id": class_id,
            "timestamp": datetime.now(),
            "date": today,
            "hour": current_period,
            "status": "Present"
        }

        if current_period is None:
            print("[⚠] No active period right now. Attendance not logged.")

        # Prevent duplicates for same student on same date and period
        if not attendance_col.find_one({
            "student_id": student_id,
            "class_id": class_id,
            "date": today,
            "hour": current_period
        }):
            attendance_col.insert_one(record)
            print(f"[✔] Marked present: {student_id} | Period {current_period}")
        else:
            print(f"[ℹ] Already marked: {student_id} | Period {current_period}")
