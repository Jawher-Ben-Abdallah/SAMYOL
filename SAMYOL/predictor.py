from samyol.yolo_preprocessing import YOLOPreProcessing
from samyol.yolo_inference import YOLOInference
from samyol.yolo_postprocessing import YOLOPostProcessing
from samyol.prediction_results import SAMYOLPredictions
from samyol.sam_inference import SAM
from samyol.utils import perform_samyol_input_checks
from typing import Union, List, Optional, Dict, Callable

class SAMYOL:   
    def __init__(
        self,
        yolo_model_path: str,
        yolo_version: str,
        sam_model_type: str,
        sam_source: str,
        class_labels: List[str],
        device: str,
        extra_args: Optional[Dict] = None,
        ):
        """
        Initialize the SAMYOL object.

        Args:
            yolo_model_path (str): Path to the YOLO model.
            yolo_version (str): Version of the YOLO model to use.
            sam_model_type (str): SAM model type to use: base, large or huge.
            sam_source (str): SAM source: HuggingFace or facebook repo.
            class_labels (List[str]): List of class labels.
            device (str): Device to use for inference.
            extra_args (Dict, optional): Extra arguments to be passed to the YOLO-NAS inference step. Defaults to None.
        """
        perform_samyol_input_checks(
            yolo_version=yolo_version,
            sam_source=sam_source,
            sam_model_type=sam_model_type,
            device=device
        )
        self.yolo_model_path = yolo_model_path
        self.yolo_version = yolo_version
        self.sam_model_type = sam_model_type
        self.sam_source = sam_source
        self.class_labels = class_labels
        self.device = device
        self.kwargs = extra_args if extra_args is not None else {}

    def predict(
            self,
            input_paths: Union[str, List[str]],
    ) -> SAMYOLPredictions:
        """
        Predicts object segmentation using the SAMYOL model.

        Args:
            input_paths (Union[str, List[str]]): Path(s) to the input image(s).

        Returns:
            SAMYOLPredictions: Object containing the input images and segmentation predictions.
        """
        if not isinstance(input_paths, List):
            input_paths = [input_paths]

        yolo_pipeline = self.get_yolo_pipeline(self.yolo_version)
        preprocessed_data = yolo_pipeline['preprocessing'](input_paths)
        outputs = yolo_pipeline['inference'](self.yolo_model_path, preprocessed_data, self.device, **self.kwargs)
        obj_det_predictions = yolo_pipeline['postprocessing'](outputs, self.class_labels)
        
        sam_pipeline = self.get_sam_pipeline(self.sam_source)
        object_segmentation_predictions = sam_pipeline['sam_inference'](self.sam_model_type, preprocessed_data[-1], obj_det_predictions, self.device)

        return SAMYOLPredictions(
            images=preprocessed_data[-1], 
            predictions=object_segmentation_predictions,
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

    @staticmethod
    def get_sam_pipeline(source: str) -> Dict[str, Callable]:
        """
        Get the SAM pipeline components based on the specified source.

        Args:
            source (str): Source for SAM model inference.

        Returns:
            Dict[str, Callable]: Dictionary containing the SAM pipeline components.
        """
        run_sam_inference = getattr(SAM, f"predict_from_{source}")
        return {
            'sam_inference': run_sam_inference
                }
