class YOLOPostProcessing():
    def __init__(self) -> None:
        print("Did a bunch of stuff")

    def get_yolo_6_postprocessing():
        print("Fetching yolo 6 postprocessing")
    
    def get_yolo_7_postprocessing():
        print("Fetching yolo 7 postprocessing")

    def get_yolo_8_postprocessing():
        print("Fetching yolo 8 postprocessing")

    def get_yolo_nas_postprocessing(detections):
        object_detection_predictions = []
        for i, detection in enumerate(detections):
            class_names = detection.class_names
            labels = detection.predction.labels
            confidence = detection.prediction.confidence
            bboxes = detection.prediction.bboxes_xyxy

            for label, conf, bbox in zip(labels, confidence, bboxes):
                object_detection_predictions.append({
                    'image_id': i,
                    'class_id': class_names[int(label)],
                    'confidence': conf,
                    'bbox': bbox
                })
        return object_detection_predictions