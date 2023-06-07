from samyol.yolo_preprocessing import YOLOPreProcessing
from samyol.yolo_inference import YOLOInference
from samyol.yolo_postprocessing import YOLOPostProcessing
from samyol.prediction_results import SAMYOLPredictions
from typing import Union, List, Optional, Dict, Tuple, Callable
from samyol.sam_inference import HuggingFaceSAMModel
import numpy as np

class SAMYOL:   
    def __init__(
        self,
        model_path: str,
        device: str,
        version: str,
        class_labels: List[str],
        extra_args: Optional[Dict] = None
    ) -> None:
        """
        Initialize the SAMYOL object.

        Args:
            model_path (str): Path to the YOLO model.
            device (str): Device to use for inference.
            version (str): Version of the YOLO model to use.
            class_labels (List[str]): List of class labels.
            extra_args (Dict, optional): Extra arguments to be passed to the YOLO-NAS inference step. Defaults to None.
        """
        self.model_path = model_path
        self.version = version
        self.class_labels = class_labels
        self.kwargs = extra_args if extra_args is not None else {}
        self.device = device

    def predict(
            self,
            input_paths: Union[str, List[str]],
        ) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Run the YOLO-based object detection pipeline followed by SAM and obtain object segmentation predictions.

        Args:
            input_paths (Union[str, List[str]]): Path(s) to the input images.

        Returns:
            Tuple[List[np.ndarray], List[Dict]]: A tuple of original RGB images and object segmentation predictions.
        """
        if not isinstance(input_paths, List):
            input_paths = [input_paths]
        yolo_pipeline = self.get_yolo_pipeline(self.version)
        preprocessed_data = yolo_pipeline['preprocessing'](input_paths)
        outputs = yolo_pipeline['inference'](self.model_path, preprocessed_data, **self.kwargs)
        obj_det_predictions = yolo_pipeline['postprocessing'](outputs)
        object_segmentation_predictions = HuggingFaceSAMModel(preprocessed_data[-1], obj_det_predictions, self.device).sam_inference()
        return SAMYOLPredictions(
            images=preprocessed_data[-1], 
            predictions=object_segmentation_predictions,
            class_labels=self.class_labels
        )

    @staticmethod
    def get_yolo_pipeline(version: str) -> Dict[str, Callable]:
        """
        Get the YOLO pipeline components based on the specified version.

        Args:
            version (str): Version of the YOLO model.

        Returns:
            Dict[str, Callable]: Dictionary containing the YOLO pipeline components.
        """
        run_yolo_preprocessing = getattr(YOLOPreProcessing, f"get_yolo_{version}_preprocessing")
        run_yolo_inference = getattr(YOLOInference, f"get_yolo_{version}_inference")
        run_yolo_postprocessing = getattr(YOLOPostProcessing, f"get_yolo_{version}_postprocessing")
        return {
            'preprocessing': run_yolo_preprocessing, 
            'inference': run_yolo_inference, 
            'postprocessing': run_yolo_postprocessing
        }