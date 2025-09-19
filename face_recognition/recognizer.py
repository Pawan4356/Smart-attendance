import os
import cv2
import numpy as np
import onnxruntime as ort
from utils.preprocessing import preprocess_face

providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
available_providers = ort.get_available_providers()

class FaceRecognizer:
    def __init__(self, model_path, mongo_handler, threshold):
        self.arcface_session = ort.InferenceSession(
            model_path,
            providers=[p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in ort.get_available_providers()]
        )
        self.input_name = self.arcface_session.get_inputs()[0].name
        self.threshold = threshold
        self.known_embeddings = {}  # student_id -> embedding
        self.student_metadata = {}  # student_id -> {class_id, name}

        # Load all student embeddings from DB
        students = mongo_handler.get_all_registered_students()
        for student in students:
            student_id = student["student_id"]
            class_id = student["class_id"]
            name = student["name"]

            raw_embs = student.get("embeddings", [])
            if not raw_embs:
                continue  # Skip students with no embeddings

            emb_array = np.array(raw_embs, dtype=np.float32)
            norm_embs = emb_array / np.linalg.norm(emb_array, axis=1, keepdims=True)

            self.known_embeddings[student_id] = norm_embs  # shape: (N, 512)
            self.student_metadata[student_id] = {
                "class_id": class_id,
                "name": name
            }


    def recognize(self, face_img):
        processed = preprocess_face(face_img)
        emb = self.arcface_session.run(None, {self.input_name: processed})[0].flatten()
        emb = emb / np.linalg.norm(emb)
        best_score = -1
        best_match = None

        for student_id, known_emb_list in self.known_embeddings.items():
            if known_emb_list.ndim == 1:
                known_emb_list = known_emb_list[np.newaxis, :]
            for known_emb in known_emb_list:
                sim = np.dot(emb, known_emb)
                if sim > best_score:
                    best_score = sim
                    best_match = student_id


        if best_score >= self.threshold and best_match in self.student_metadata:
            meta = self.student_metadata[best_match]
            return best_match, meta["class_id"], meta["name"], best_score
        return None

    
    def get_embedding(self, face_img):
        """
        Takes a single face image (BGR), returns 512-dim normalized embedding.
        """
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (112, 112))
        face_norm = (face_resized - 127.5) * 0.0078125  # Scale to [-1, 1]
        face_input = np.transpose(face_norm, (2, 0, 1))[None].astype(np.float32)

        embedding = self.arcface_session.run(None, {self.input_name: face_input})[0].flatten()
        embedding /= np.linalg.norm(embedding)
        return embedding.astype(np.float32)
