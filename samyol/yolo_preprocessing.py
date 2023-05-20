from .utils import generic_yolo_preprocessing
from typing import List, Tuple, Union
import numpy as np
class YOLOPreProcessing():
    
    @staticmethod
    def get_yolo_6_preprocessing(
        inputs: List[str]
        ) -> Tuple[np.ndarray, List[Tuple[np.ndarray, float, Tuple[float, float]]], List[np.ndarray]]:
        """
        Perform YOLO-6 preprocessing on the inputs.

        Args:
            inputs (List[str]): List of input image paths.

        Returns:
            Tuple[np.ndarray, List[Tuple[np.ndarray, float, Tuple[float, float]]], List[np.ndarray]]: Preprocessed data, including the batch of resized images, ratio and padding information for each image, and the original RGB images.
        """
        return generic_yolo_preprocessing(inputs)

    @staticmethod
    def get_yolo_7_preprocessing(
        inputs: List[str]
        ) -> Tuple[np.ndarray, List[Tuple[np.ndarray, float, Tuple[float, float]]], List[np.ndarray]]: 
        """
        Perform YOLO-7 preprocessing on the inputs.

        Args:
            inputs (List[str]): List of input image paths.

        Returns:
            Tuple[np.ndarray, List[Tuple[np.ndarray, float, Tuple[float, float]]], List[np.ndarray]]: Preprocessed data, including the batch of resized images, ratio and padding information for each image, and the original RGB images.
        """
        return generic_yolo_preprocessing(inputs)

    @staticmethod
    def get_yolo_8_preprocessing(
        inputs: Union[List[str], List[np.ndarray]]
        ) -> Union[List[str], List[np.ndarray]]:
        """
        Perform YOLOv8 preprocessing on the inputs.

        Args:
            inputs (Union[List[str], List[np.ndarray]]): List of input image paths or list of input images.

        Returns:
            inputs (Union[List[str], List[np.ndarray]]): List of input image paths or list of input images.
        """
        return inputs
    
    @staticmethod
    def get_yolo_nas_preprocessing(
        inputs: Union[List[str], List[np.ndarray]]
        ) -> Union[List[str], List[np.ndarray]]:
        """
        Perform YOLO-NAS preprocessing on the inputs.

        Args:
            inputs (Union[List[str], List[np.ndarray]]): List of input image paths or list of input images.

        Returns:
            inputs (Union[List[str], List[np.ndarray]]): List of input image paths or list of input images.
        """
        return inputs