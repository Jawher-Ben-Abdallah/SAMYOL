import numpy as np
from ultralytics import YOLO


def run_object_detection_inference(weights_path, image):
    model = YOLO(weights_path)
    detections = model.predict(
        image, 
        conf=0.35, 
        iou=0.65, 
        verbose=False
        )
    return detections


def run_object_detection_postprocess(detections):
    object_detection_predictions = []
    for i, detection in enumerate(detections):
        class_labels = detection.names
        boxes = detection.boxes
        for box in boxes:
            object_detection_predictions.append(
                {
                    'image_id': i,
                    'class_id': class_labels[int(box.cls.item())],
                    'bbox': box.xyxy[0].numpy().round().astype(np.int32).tolist()
                }
            )
    return object_detection_predictions


def get_postprocessed_detections(weights_path, image):
    detections = run_object_detection_inference(weights_path, image)
    detections = run_object_detection_postprocess(detections)
    return detections