# 🎓 Face Recognition Attendance System

A real-time face recognition-based attendance system using **YOLOv8/YOLOv11n** for face detection, **ArcFace (ONNX)** for face embeddings, and **FAISS** for fast and scalable face matching. The system captures frames from a **WiFi camera**, recognizes students, and logs their attendance in a **CSV file**, while storing metadata in **MongoDB**.

## 📌 Features

- 🔍 **Face Detection** using YOLOv8/YOLOv11n
- 🧠 **Face Recognition** using ArcFace embeddings
- 📷 **Live Frame Capture** from WiFi camera
- 📦 **Student Data** stored in MongoDB (name, ID, class)
- 🧾 **Attendance Logging** in CSV format
- 🚀 Fast search with FAISS index


## 🛠️ Tech Stack

| Component        | Technology             |
|------------------|------------------------|
| Face Detection   | YOLOv8 / YOLOv11n      |
| Face Embeddings  | ArcFace (ONNX)         |
| Face Search      | FAISS                  |
| Backend Language | Python                 |
| Database         | MongoDB                |
| Logging Format   | MongoDB                |
| Video Source     | WiFi Camera (IP-based) |

## 📂 Project Structure

```bash
project/
├── camera_capture/
│   └── capture.py
├── face_detection/
│   └── detect_faces.py
├── face_recognition/
│   ├── arcface_model.py
│   └── recognize.py
├── attendance_log/
│   └── log_attendance.py
├── database/
│   └── students.db
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

### 4. Configure MongoDB

Install Both models Required.

### 5. Run the System

```bash
python main.py
```
