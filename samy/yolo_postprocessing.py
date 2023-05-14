import cv2
from PIL import Image
import numpy as np
from utils import letterbox


class YOLOPostProcessing():

    @staticmethod
    def get_yolo_6_postprocessing(outputs):
        object_detection_predictions = []
        detections, rezise_data, origin_RGB = outputs
        for i in range(detections[0].shape[0]):
            obj_num = detections[0][i]
            boxes = detections[1][i]
            scores = detections[2][i]
            cls_id = detections[3][i]
            image = origin_RGB[i]
            img_h, img_w = image.shape[:2]
            ratio, dwdh = rezise_data[i][1:]
            for num in range(obj_num[0]):
                box = boxes[num]
                score = round(float(scores[num]), 3)
                class_id = int(cls_id[num])
                box -= np.array(dwdh*2)
                box /= ratio
                box = box.round().astype(np.int32).tolist()
                x1 = max(0, box[0])
                y1 = max(0, box[1])
                x2 = min(img_w, box[2])
                y2 = min(img_h, box[3])
                object_detection_predictions.append({
                    'image_id': i,
                    'class_id': class_id,
                    'score': score,
                    'bbox': [x1, y1, x2, y2]
                })
        return object_detection_predictions


    
    def get_yolo_7_postprocessing(img, detections, class_labels): #img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = img.copy()
        object_detection_predictions = []
        image, ratio, dwdh = letterbox(image, auto=False)
        
        for i,(batch_id, x0, y0, x1, y1, cls_id, score) in enumerate(detections):
            score = round(float(score),3)
            box = np.array([x0,y0,x1,y1])
            box -= np.array(dwdh*2)
            box /= ratio
            object_detection_predictions.append(
                {
                    'image_id': int(batch_id),
                    'class_id': int(cls_id),
                    'score': score,
                    'bbox': box.round().astype(np.int32).tolist()
                }
            )
        return object_detection_predictions
    
    
    def get_yolo_8_postprocessing(self, detections):
        object_detection_predictions = []
        for i, detection in enumerate(detections):
            class_names = detection.names
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


    def get_yolo_nas_postprocessing(self, detections):
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