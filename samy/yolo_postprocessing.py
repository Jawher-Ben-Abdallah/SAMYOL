import cv2
import numpy as np
from PIL import Image
from utils import letterbox




class YOLOPostProcessing():
    def __init__(self) -> None:
        print("Did a bunch of stuff")

    def get_yolo_6_postprocessing():
        print("Fetching yolo 6 postprocessing")
    
    def get_yolo_7_postprocessing(img, detections, class_labels): #img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = img.copy()
        image, ratio, dwdh = letterbox(image, auto=False)

        object_detection_predictions = []

        for i,(batch_id ,x0,y0,x1,y1,cls_id,score) in enumerate(detections):
            score = round(float(score),3)

            box = np.array([x0,y0,x1,y1])
            box -= np.array(dwdh*2)
            box /= ratio
            
            object_detection_predictions.append(
                {
                    'image_id': int(batch_id),
                    'class_id': class_labels[int(cls_id)],
                    'score': score,
                    'bbox': box.round().astype(np.int32).tolist()
                }
            )

        return object_detection_predictions
    
    
    def get_yolo_8_postprocessing(detections):
        object_detection_predictions = []
        for i, detection in enumerate(detections):
            class_names = detection.class_names
            boxes = detection.boxes 

            for box in boxes:
                object_detection_predictions.append(
                    {
                        'image_id': i,
                        'class_id': class_names[int(box.cls.item())],
                        'score': box.conf.item(),
                        'bbox': box.xyxy[0].numpy().round().astype(np.int32).tolist()
                    }
                )
        return object_detection_predictions


    def get_yolo_nas_postprocessing(detections):
        object_detection_predictions = []
        for i, detection in enumerate(detections):
            class_names = detection.class_names
            labels = detection.prediction.labels
            confidence = detection.prediction.confidence
            bboxes = detection.prediction.bboxes_xyxy

            for label, conf, bbox in zip(labels, confidence, bboxes):
                object_detection_predictions.append({
                    'image_id': i,
                    'class_id': class_names[int(label)],
                    'confidence': conf,
                    'bbox': bbox.round().astype(np.int32).tolist()
                })
        return object_detection_predictions