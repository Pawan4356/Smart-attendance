import os
import cv2
import warnings
import numpy as np
from database.mongo_utils import insert_student
from face_recognition.face_alignment import align_face
from face_recognition.arcface_model import get_arcface_embedding

AUG_PER_IMAGE = 5  # How many augmented embeddings to generate per student
MAX_EMBEDDINGS = 6

def augment_image(img):
    augmented = []

    # Horizontal Flip
    augmented.append(cv2.flip(img, 1))

    # Brightness/Contrast
    alpha = 1 + 0.2 * (np.random.rand() - 0.5)
    beta = 20 * (np.random.rand() - 0.5) * 2
    augmented.append(cv2.convertScaleAbs(img, alpha=alpha, beta=beta))

    # Rotation
    h, w = img.shape[:2]
    angle = np.random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    rotated = cv2.warpAffine(img, M, (w, h))
    augmented.append(rotated)

    # Gaussian Blur
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    augmented.append(blurred)

    # Add Noise
    noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
    noisy = cv2.add(img, noise)
    augmented.append(noisy)

    return augmented[:AUG_PER_IMAGE]

from database.mongo_utils import get_student_by_id  # ADD THIS IMPORT

def enroll_students_from_folder(folder_path="./faces"):
    """
    Enroll students by processing image files from a folder.
    Skips if student already has enough embeddings stored.
    """
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        try:
            name_without_ext = os.path.splitext(filename)[0]
            parts = name_without_ext.split("_", 2)

            if len(parts) < 3:
                print(f"[SKIP] Invalid format: {filename}")
                continue

            class_id, student_id, name_part = parts
            name = name_part.replace("_", " ")

            # Check existing data
            existing = get_student_by_id(class_id, student_id)
            if existing:
                stored_embs = existing.get("embeddings", []) or [existing.get("embedding")]
                if stored_embs and len(stored_embs) >= MAX_EMBEDDINGS:
                    print(f"[ℹ] Skipping {student_id} — already has {len(stored_embs)} embeddings.")
                    continue

            image_path = os.path.join(folder_path, filename)
            img = cv2.imread(image_path)
            if img is None:
                print(f"[ERROR] Couldn't read image: {filename}")
                continue

            img = align_face(img)

            # Augment + Embed
            images = [img] + augment_image(img)
            embeddings = []
            for im in images:
                try:
                    emb = get_arcface_embedding(im)
                    embeddings.append(emb)
                except Exception as emb_err:
                    warnings.warn(f"Embedding failed for {filename}: {emb_err}")

            if embeddings:
                insert_student(student_id, name, class_id, embeddings)
                print(f"[OK] Enrolled: {name} ({student_id}) with {len(embeddings)} embeddings")
            else:
                print(f"[SKIP] No valid embeddings for {student_id}")

        except Exception as e:
            print(f"[ERROR] Failed to process {filename}: {e}")
