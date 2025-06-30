import os
import cv2
import warnings
import numpy as np
from database.mongo_utils import MongoHandler
from face_recognition.recognizer import FaceRecognizer

AUG_PER_IMAGE = 5

def augment_image(img, base_path, base_name):
    augmented = []

    def save_augmented(im, suffix):
        path = os.path.join(base_path, f"{base_name}_{suffix}.jpg")
        if not os.path.exists(path):
            cv2.imwrite(path, im)
            return im
        return None

    # Horizontal Flip
    flip = cv2.flip(img, 1)
    if (aug := save_augmented(flip, "flip")) is not None:
        augmented.append(aug)

    # CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    clahe_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    if (aug := save_augmented(clahe_img, "clahe")) is not None:
        augmented.append(aug)

    # Rotation
    h, w = img.shape[:2]
    angle = np.random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    rotated = cv2.warpAffine(img, M, (w, h))
    if (aug := save_augmented(rotated, "rotate")) is not None:
        augmented.append(aug)

    # Gaussian Blur
    blurred = cv2.GaussianBlur(img, (5, 11), 0)
    if (aug := save_augmented(blurred, "blur")) is not None:
        augmented.append(aug)

    # Add Noise
    noise = np.random.normal(0, 0.6, img.shape).astype(np.uint8)
    noisy = cv2.add(img, noise)
    if (aug := save_augmented(noisy, "noise")) is not None:
        augmented.append(aug)

    return augmented[:AUG_PER_IMAGE]


def enroll_students(class_id, threshold, uri, base_folder="./faces"):
    """
    Enroll students from subfolders in `faces/`, where each folder = student_id.
    Only generate augmented images if not already present.
    """

    for student_folder in os.listdir(base_folder):
        student_path = os.path.join(base_folder, student_folder)
        if not os.path.isdir(student_path):
            continue

        try:
            parts = student_folder.split("_", 2)
            if len(parts) < 3:
                print(f"Invalid folder format: {student_folder}")
                continue

            class_id, student_id, name_part = parts
            name = name_part.replace("_", " ")

            mongo_handler = MongoHandler(uri, class_id)
            face_recognizer = FaceRecognizer("models/w600k_r50.onnx", mongo_handler, threshold)

            existing = mongo_handler.get_student_by_id(class_id, student_id)
            if existing:
                print(f"Skipping {student_id} — already enrolled.")
                continue


            # Load all base images (excluding augmented ones)
            base_imgs = [f for f in os.listdir(student_path)
                        if not any(x in f for x in ["_blur", "_noise", "_clahe", "_flip", "_rotate"])
                        and f.lower().endswith((".jpg", ".png"))]

            if not base_imgs:
                print(f"No base images found for {student_id}")
                continue

            # Augment and save if not already present
            for img_name in base_imgs:
                base_img_path = os.path.join(student_path, img_name)
                base_img = cv2.imread(base_img_path)
                if base_img is None:
                    print(f"Failed to read base image: {img_name}")
                    continue

                base_name = os.path.splitext(img_name)[0]
                augment_image(base_img, student_path, base_name)


            # Load all images including augmented
            all_imgs = [cv2.imread(os.path.join(student_path, img))
                        for img in os.listdir(student_path)
                        if img.lower().endswith((".jpg", ".png"))]

            embeddings = []
            for im in all_imgs:
                if im is not None:
                    try:
                        emb = face_recognizer.get_embedding(im)
                        emb = emb / np.linalg.norm(emb)
                        embeddings.append(emb)
                    except Exception as emb_err:
                        warnings.warn(f"Embedding failed in {student_folder}: {emb_err}")

            if embeddings:
                mongo_handler.insert_student(student_id, name, class_id, embeddings)
                print(f"Enrolled {name} ({student_id}) with {len(embeddings)} embeddings")
            else:
                print(f"No valid embeddings for {student_id}")

        except Exception as e:
            print(f"Error in folder {student_folder}: {e}")
