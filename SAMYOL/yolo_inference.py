import cv2
import torch
from typing import List, Tuple
from pathlib import Path

from .utils import generic_ort_inference, check_and_install_library


class YOLOInference():
    @staticmethod
    def get_yolo_4_inference(model_weights: str, model_cfg: str, inputs: List) -> List:
        """
        Perform YOLO-4 inference.

        Args:
            model_weights (str): Path to the YOLOv4 model weights.
            model_cfg (str): Path to the YOLOv4 model config file.
            inputs (List): Input data.

        Returns:
            detections (List): List of detection results.
        """
        net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
        model = cv2.dnn_DetectionModel(net)
        model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=False)
        detections = [model.detect(image, confThreshold=0.6, nmsThreshold=0.4) for image in inputs]
        return detections

    @staticmethod
    def get_yolo_6_inference(model_path: str, inputs: Tuple, device: str) -> Tuple:
        """
        Perform YOLO-6 inference.

        Args:
            model_path (str): Path to the YOLOv6 model.
            inputs (Tuple): Input data.
            device (str): Device to use for inference.

        Returns:
            Tuple: Inference results, resize data, and original RGB images.
        """
        np_batch, resize_data, origin_RGB = inputs
        cuda = True if device == "cuda" else False
        detections = generic_ort_inference(model_path, np_batch, cuda=cuda)
        return detections, resize_data, origin_RGB
    
    @staticmethod
    def get_yolo_7_inference(model_path: str, inputs: Tuple, device:str) -> Tuple:
        """
        Perform YOLO-7 inference.

        Args:
            model_path (str): Path to the YOLOv7 model.
            inputs (Tuple): Input data.
            device (str): Device to use for inference.

        Returns:
            Tuple: Inference results, resize data, and original RGB images.
        """
        np_batch, resize_data, origin_RGB = inputs
        cuda = True if device == "cuda" else False
        detections = generic_ort_inference(model_path, np_batch, cuda=cuda)
        return detections, resize_data, origin_RGB

    @staticmethod
    def get_yolo_8_inference(model_path: str, inputs: List, device:str) -> List:
        """
        Perform YOLO-8 inference.

        Args:
            model_path (str): Path to the YOLOv8 model.
            inputs (Tuple): Input data.
            device (str): Device to use for inference.

        Returns:
            Tuple: Inference results, resize data, and original RGB images.
        """
        check_and_install_library('ultralytics')
        from ultralytics import YOLO
        model = YOLO(model_path)
        detections = [model.predict(
            image, 
            conf=0.35, 
            iou=0.65, 
            verbose=False, 
            device=str(torch.cuda.current_device()) if device=="cuda" else "cpu"
        ) for image in inputs[0]]
        return detections

    
    @staticmethod
    def get_yolo_nas_inference(
        model_path: str, 
        inputs: List,
        device:str,
        model_type: str, 
        num_classes: int
        ) -> List:
        """
        Perform YOLO-NAS inference.

        Args:
            model_path (str): Path to the YOLO-NAS model.
            inputs (List): Input data.
            device (str): Device to use for inference.
            model_type (str): Model type.
            num_classes (int): Number of classes.

        Returns:
            List: List of detection results.
        """
        check_and_install_library('super_gradients')
        import super_gradients as sg
        from super_gradients.training import models
        sg.setup_device(device=device)
        model = models.get(
            model_type,
            num_classes=num_classes,
            checkpoint_path=model_path
        ).to(torch.device(device))
        detections = model.predict(inputs[0])
        return detections