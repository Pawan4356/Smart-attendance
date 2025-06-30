import cv2
import numpy as np

def preprocess_face(face_img):
    rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (112, 112))
    normalized = (resized - 127.5) * 0.0078125
    transposed = np.transpose(normalized, (2, 0, 1))
    return np.expand_dims(transposed, axis=0).astype(np.float32)
