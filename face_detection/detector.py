import cv2
import torch
from ultralytics import YOLO

class FaceDetector:
    def __init__(self, model_path: str):
        # Load YOLO face detection model
        self.model = YOLO(model_path)
        print(f"Loaded YOLO model from {model_path}. Using CUDA: {torch.cuda.is_available()}")

    def detect_faces(self, image, conf_threshold: float = 0.5):
        """
        Detect faces in an image using YOLO model.
        Returns list of bounding boxes (x1, y1, x2, y2) and confidences for detections above conf_threshold.
        """
        results = self.model(image)
        boxes = []
        confs = []
        # Check if any results with detected boxes
        if results and len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            bboxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            for bbox, score in zip(bboxes, scores):
                if score >= conf_threshold:
                    x1, y1, x2, y2 = bbox.astype(int)
                    # Clip box coordinates to image dimensions
                    h, w = image.shape[:2]
                    x1 = max(0, min(x1, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    x2 = max(0, min(x2, w - 1))
                    y2 = max(0, min(y2, h - 1))
                    boxes.append((x1, y1, x2, y2))
                    confs.append(float(score))
        return boxes, confs
