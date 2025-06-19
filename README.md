# 🎓 Face Recognition Attendance System

A real-time face recognition-based attendance system using **YOLOv8** for face detection and **ArcFace (ONNX)** for face embeddings and scalable face matching. The system captures frames from a **WiFi camera**, recognizes students, and logs their attendance in the **Database** and stores metadata as well.

## 📌 Features

- 🔍 **Face Detection** using YOLOv8
- 🧠 **Face Recognition** using ArcFace embeddings
- 📷 **Live Frame Capture** from WiFi camera
- 📦 **Student Data** stored in MongoDB (name, ID, class)
- 🧾 **Attendance Logging** in MongoDB


## 🛠️ Tech Stack

| Component        | Technology             |
|------------------|------------------------|
| Face Detection   | YOLOv8 / YOLOv11n      |
| Face Embeddings  | ArcFace (ONNX)         |
| Backend Language | Python                 |
| Database         | MongoDB                |
| Logging Format   | MongoDB                |
| Video Source     | WiFi Camera (IP-based) |

## 📂 Project Structure

```bash
project/
├── camera_capture/
│   └── capture.py
├── captures/
├── face_detection/
│   └── detect_faces.py
├── face_recognition/
│   ├── arcface_model.py
│   └── recognize.py
├── attendance_log/
│   └── log_attendance.py
├── database/
│   ├── enroll_student.py
│   └── mongo_utils.py
├── training/
│   └── fine_tune.py
└── main.py
```

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/attendance-system.git
cd attendance-system
```

### 2. Create Virtual Environment & Install Dependencies

```bash
python -m venv attendance
source attendance/bin/activate  # On Windows: attendance\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure MongoDB

Make sure MongoDB is running. Update connection URI and collection info in ***database*** folder.

### 4. Run the System

```bash
python main.py
```
