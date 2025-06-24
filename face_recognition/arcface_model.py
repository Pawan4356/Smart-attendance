import os
import cv2
import numpy as np
import onnxruntime as ort

# Load ArcFace ONNX model
model_path = "models/w600k_r50.onnx"
assert os.path.exists(model_path), "ArcFace ONNX model not found at models/w600k_r50.onnx"
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
available_providers = ort.get_available_providers()

# Prefer GPU if available, else fallback to CPU
arcface_session = ort.InferenceSession(
    model_path,
    providers=[p for p in providers if p in available_providers]
)
input_name = arcface_session.get_inputs()[0].name

def get_arcface_embedding(face_img):
    """
    Takes a single face image (BGR), returns 512-dim normalized embedding.
    """
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (112, 112))
    face_norm = (face_resized - 127.5) * 0.0078125  # Scale to [-1, 1]
    face_input = np.transpose(face_norm, (2, 0, 1))[None].astype(np.float32)

    embedding = arcface_session.run(None, {input_name: face_input})[0].flatten()
    embedding /= np.linalg.norm(embedding)
    return embedding.astype(np.float32)
