import cv2
import numpy as np
import face_recognition

def align_face(img):
    face_locations = face_recognition.face_locations(img)
    face_landmarks = face_recognition.face_landmarks(img)

    if not face_landmarks:
        return img  # return as-is if no landmarks

    landmarks = face_landmarks[0]
    left_eye = np.mean(landmarks["left_eye"], axis=0)
    right_eye = np.mean(landmarks["right_eye"], axis=0)

    # compute angle
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    # rotate image to align eyes horizontally
    eyes_center = tuple(((left_eye + right_eye) / 2).astype(int))
    rot_mat = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
    aligned = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return aligned
