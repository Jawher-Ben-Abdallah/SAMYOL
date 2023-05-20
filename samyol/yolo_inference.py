import subprocess
from .utils import generic_ort_inference
import onnxruntime as ort
from .utils import generic_ort_inference
from typing import List, Tuple


class YOLOInference():

    @staticmethod
    def get_yolo_6_inference(model_path: str, inputs: Tuple) -> Tuple:
        """
        Perform YOLO-6 inference.

        Args:
            model_path (str): Path to the YOLOv6 model.
            inputs (Tuple): Input data.

        Returns:
            Tuple: Inference results, resize data, and original RGB images.
        """
        detections = generic_ort_inference(model_path, inputs, cuda=True)
        resize_data, origin_RGB = inputs[1:]
        return detections, resize_data, origin_RGB
    
    @staticmethod
    def get_yolo_7_inference(model_path: str, inputs: Tuple) -> Tuple:
        """
        Perform YOLO-7 inference.

        Args:
            model_path (str): Path to the YOLOv7 model.
            inputs (Tuple): Input data.

        Returns:
            Tuple: Inference results, resize data, and original RGB images.
        """
        detections = generic_ort_inference(model_path, inputs, cuda=True)
        resize_data, origin_RGB = inputs[1:]
        return detections, resize_data, origin_RGB

    @staticmethod
    def get_yolo_8_inference(model_path: str, inputs: List) -> List:
        """
        Perform YOLO-8 inference.

        Args:
            model_path (str): Path to the YOLOv8 model.
            inputs (Tuple): Input data.

        Returns:
            Tuple: Inference results, resize data, and original RGB images.
        """

        try:
            from ultralytics import YOLO
        except ImportError:
            print('Installing ultralytics ...')
            subprocess.check_call(["python", '-m', 'pip', 'install', 'ultralytics'])
        
        model = YOLO(model_path)
        detections = []
        for image in inputs:
            detections.append(
                model.predict(
                    image, 
                    conf=0.35, 
                    iou=0.65, 
                    verbose=False
                    )
                )
        return detections

    
    @staticmethod
    def get_yolo_nas_inference(
        model_path: str, 
        inputs: List, 
        model_type: str, 
        num_classes: int
        ) -> List:
        """
        Perform YOLO-NAS inference.

        Args:
            model_path (str): Path to the YOLO-NAS model.
            inputs (List): Input data.
            model_type (str): Model type.
            num_classes (int): Number of classes.

        Returns:
            List: List of detection results.
        """
        try:
            from super_gradients.training import models
        except ImportError:
            print('Installing super-gradients ...')
            subprocess.check_call(["python", '-m', 'pip', 'install', 'super-gradients'])
        
        model = models.get(
            model_type,
            num_classes=num_classes,
            checkpoint_path=model_path
        )
        detections = model.predict(inputs)
        return detections