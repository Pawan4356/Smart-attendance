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

### _capture.py_
#### Purpose:
- Controls the camera to capture an image of the classroom.
#### Key Features:
- Handles webcam/video stream
- Captures and stores a classroom snapshot
### _detect_face.py_
#### Purpose:
 - Detects faces from the image captured by capture.py and pre-processes them.
#### Key Features:
- Uses a face detection model (e.g. YOLO)
- Crops face regions and performs preprocessing (resize, normalize)
### _arcface_model.py_
#### Purpose:
 - Generates embeddings for the preprocessed faces using the ArcFace model.
#### Key Features:
- Loads ONNX ArcFace model
- Converts faces to 512-D embeddings
- Outputs a list of embeddings
### _recognize.py_
#### Purpose:
 - Matches face embeddings to known identities using FAISS.
#### Key Features:
- Builds a FAISS index from enrolled student embeddings
- Implements recognize_faces() to match faces
- Returns a list of recognized student_ids 
### _log_attendance.py_
#### Purpose:
 - Logs attendance based on recognized student_ids.
#### Key Features:
- Marks students present in their respective class
- Optionally updates timestamp or session info
### _mongo_utils.py_
#### Purpose:
 - Provides database utilities and operations.
#### Key Features:
- Connects to MongoDB
- Inserts, updates, and queries student and attendance data
- Handles schema-related functions
### _enroll_student.py_
#### Purpose:
 - Enrolls students into the database using images in the faces/ directory.
#### Key Features:
- Iterates through student folders/images
- Uses arcface_model.py to get embeddings
- Stores student info and embeddings in MongoDB
### _fine_tune.py_
#### Purpose:
- Periodically fine-tunes the system using collected face data.
#### Key Features:
- Aggregates new face embeddings over time
- Computes average embeddings
- Updates stored vectors for improved accuracy

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
