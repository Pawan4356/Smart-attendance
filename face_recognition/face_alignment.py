import cv2
import numpy as np
import face_alignment
from skimage import io

fa = face_alignment.FaceAlignment('2D', flip_input=False, device='cpu')

def align_face(img):
    preds = fa.get_landmarks(img)
    if preds is None:
        return img

    landmarks = preds[0]
    left_eye = np.mean(landmarks[36:42], axis=0)  # left eye landmarks
    right_eye = np.mean(landmarks[42:48], axis=0)

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    # eyes_center = tuple(((left_eye + right_eye) / 2).astype(int))
    center = (left_eye + right_eye) / 2
    eyes_center = (float(center[0]), float(center[1]))


    rot_mat = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
    aligned = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return aligned
