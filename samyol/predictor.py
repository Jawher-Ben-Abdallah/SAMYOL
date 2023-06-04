from samyol.yolo_preprocessing import YOLOPreProcessing
from samyol.yolo_inference import YOLOInference
from samyol.yolo_postprocessing import YOLOPostProcessing
from typing import Union, List, Optional, Dict, Tuple, Callable
from samyol.sam_inference import HuggingFaceSAMModel

class SAMYOL:   
    def __init__(
        self,
        model_path: str,
        device: str,
        version: str,
        extra_args: Optional[Dict] = None
    ) -> None:
        """
        Initialize the SAMYOL object.

        Args:
            model_path (str): Path to the YOLO model.
            device (str): Device to use for inference.
            version (str): Version of the YOLO model to use.
            extra_args (Dict, optional): Extra arguments to be passed to the YOLO-NAS inference step. Defaults to None.
        """
        self.model_path = model_path
        self.version = version
        self.kwargs = extra_args if extra_args is not None else {}
        self.device = device

    def predict(
            self,
            input_paths: Union[str, List[str]],
        ) -> Tuple[List, List]:
        """
        Run the YOLO-based object detection pipeline and obtain object detection predictions.

        Args:
            input_paths (Union[str, List[str]]): Path(s) to the input images.

        Returns:
            Tuple[List, List]: Object detection predictions as a tuple of masks and scores.
        """
        if not isinstance(input_paths, List):
            input_paths = [input_paths]
        yolo_pipeline = self.get_yolo_pipeline(self.version)
        preprocessed_data = yolo_pipeline['preprocessing'](input_paths)
        outputs = yolo_pipeline['inference'](self.model_path, preprocessed_data, **self.kwargs)
        obj_det_predictions = yolo_pipeline['postprocessing'](outputs)
        bbox = obj_det_predictions['bbox']
        masks, scores = HuggingFaceSAMModel(input_paths, bbox).sam_inference(self.device)
        return masks, scores
    
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