import numpy as np
from typing import List

class YOLOPostProcessing():

    @staticmethod
    def get_yolo_6_postprocessing(detections:tuple, class_labels: List[str]) -> List[dict]:
        """
        Perform YOLOv6 post-processing on the detections.

        Args:
            detections (tuple): Output data from the YOLOv6 model.
            class_labels (List[str]): List of class labels.

        Returns:
            List[dict]: List of object detection predictions.
        """
        object_detection_predictions = []
        detections, rezise_data, origin_RGB = detections
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
                    'class_label': class_labels[class_id],
                    'class_id': class_id,
                    'score': score,
                    'bbox': [x1, y1, x2, y2]
                })
        return object_detection_predictions


    @staticmethod
    def get_yolo_7_postprocessing(detections: tuple, class_labels: List[str]) -> List[dict]:
        """
        Perform YOLOv7 post-processing on the detections.

        Args:
            detections (tuple): Output data from the YOLOv7 model.
            class_labels (List[str]): List of class labels.

        Returns:
            List[dict]: List of object detection predictions.
        """
        object_detection_predictions = []
        detections, rezise_data, _ = detections
        for batch_id, x0, y0, x1, y1, cls_id, score in detections[0]:
            ratio, dwdh = rezise_data[int(batch_id)][1:]
            box = np.array([x0, y0, x1, y1])
            box -= np.array(dwdh*2)
            box /= ratio
            box = box.round().astype(np.int32).tolist()
            cls_id = int(cls_id)
            score = round(float(score),3)
            object_detection_predictions.append({
                    'image_id': int(batch_id),
                    'class_label': class_labels[cls_id],
                    'class_id': cls_id,
                    'score': score,
                    'bbox': box
                })
        return object_detection_predictions
    
    @staticmethod
    def get_yolo_8_postprocessing(detections: List[object], class_labels: List[str]) -> List[dict]:
        """
        Perform YOLOv8 post-processing on the detections.

        Args:
            detections (List[object]): Detections from the YOLOv8 model.
            class_labels (List[str]): List of class labels.

        Returns:
            List[dict]: List of object detection predictions.
        """
        object_detection_predictions = []
        for i, detection in enumerate(detections):
            boxes = detection[0].boxes
            for box in boxes:
                class_id = int(box.cls.item())
                object_detection_predictions.append(
                    {
                        'image_id': i,
                        'class_label': class_labels[class_id],
                        'class_id': class_id,
                        'score': box.conf.item(),
                        'bbox': box.xyxy[0].numpy().round().astype(np.int32).tolist()
                    }
                )
        return object_detection_predictions


    @staticmethod
    def get_yolo_nas_postprocessing(detections: List[object], class_labels: List[str]) -> List[dict]:
        """
        Perform YOLO-NAS post-processing on the detections.

        Args:
            detections (List[object]): Detections from the YOLO-NAS model.
            class_labels (List[str]): List of class labels.

        Returns:
            List[dict]: List of object detection predictions.
        """
        object_detection_predictions = []
        for i, detection in enumerate(detections):
            class_names = detection.class_names
            labels = detection.prediction.labels
            confidence = detection.prediction.confidence
            bboxes = detection.prediction.bboxes_xyxy

            for label, conf, bbox in zip(labels, confidence, bboxes):
                object_detection_predictions.append({
                    'image_id': i,
                    'class_label': class_labels[int(label)],
                    'class_id': int(label),
                    'confidence': conf,
                    'bbox': bbox.round().astype(np.int32).tolist()
                })
        return object_detection_predictions