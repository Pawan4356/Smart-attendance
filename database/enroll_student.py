import os
import cv2
import numpy as np
from database.mongo_utils import insert_student
# from face_recognition.arcface_model import get_arcface_embedding  # Uncomment if using actual embeddings

def enroll_students_from_folder(folder_path="./faces", use_temp_embedding=True):
    """
    Enroll students by processing image files from a folder.
    Filenames should follow the format: ClassID_StudentID_Name_With_Underscores.jpg

    Args:
        folder_path (str): Path to the folder containing student images.
        use_temp_embedding (bool): If True, use random 512-dim embeddings.
    """
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        try:
            # Parse filename
            name_without_ext = os.path.splitext(filename)[0]
            parts = name_without_ext.split("_", 2)

            if len(parts) < 3:
                print(f"[SKIP] Invalid format: {filename}")
                continue

            class_id, student_id, name_part = parts
            name = name_part.replace("_", " ")

            image_path = os.path.join(folder_path, filename)
            img = cv2.imread(image_path)

            if img is None:
                print(f"[ERROR] Couldn't read image: {filename}")
                continue

            if use_temp_embedding:
                embedding = np.random.rand(5).astype(np.float32)
            else:
                # embedding = get_arcface_embedding(img)  # Uncomment when real model is available
                raise NotImplementedError("ArcFace embedding not implemented yet")

            insert_student(student_id, name, class_id, embedding)
            print(f"[OK] Enrolled: {name} ({student_id})")

        except Exception as e:
            print(f"[ERROR] Failed to process {filename}: {e}")
