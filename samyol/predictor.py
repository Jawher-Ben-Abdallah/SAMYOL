from samyol.yolo_preprocessing import YOLOPreProcessing
from samyol.yolo_inference import YOLOInference
from samyol.yolo_postprocessing import YOLOPostProcessing
from typing import Union, List, Optional, Dict, Tuple, Callable
from samyol.sam_inference import HuggingFaceSAMModel
import matplotlib.pyplot as plt

class SAMYOL:   
    def __init__(
        self,
        input_paths: Union[str, List[str]],
        model_path: str,
        device: str,
        version: str,
        extra_args: Optional[Dict] = None
    ) -> None:
        """
        Initialize the SAMYOL object.

        Args:
            input_paths (Union[str, List[str]]): Path(s) to the input images.
            model_path (str): Path to the YOLO model.
            device (str): Device to use for inference.
            version (str): Version of the YOLO model to use.
            extra_args (Dict, optional): Extra arguments to be passed to the YOLO-NAS inference step. Defaults to None.
        """
        self.input_paths = input_paths
        self.model_path = model_path
        self.version = version
        self.kwargs = extra_args if extra_args is not None else {}
        self.device = device

    def predict(self) -> Tuple[List, List]:
        """
        Run the YOLO-based object detection pipeline and obtain object detection predictions.

        Returns:
            Tuple[List, List]: Object detection predictions as a tuple of masks and scores.
        """
        yolo_pipeline = self.get_yolo_pipeline(self.version)
        preprocessed_data = yolo_pipeline['preprocessing'](self.input_paths)
        outputs = yolo_pipeline['inference'](self.model_path, preprocessed_data, **self.kwargs)
        obj_det_predictions = yolo_pipeline['postprocessing'](outputs)




        masks, scores = HuggingFaceSAMModel(self.input_paths, obj_det_predictions).sam_inference(self.device)
        return preprocessed_data, masks, bboxes, scores
    

    def display(self):
        """
        Display the bounding boxes and masks.
        """
        preprocessed_data, masks, bbox, scores = self.predict()
        num_images = len(preprocessed_data)

        # Define the number of rows and columns for the subplots
        num_rows = int(num_images / 3) + (num_images % 3 > 0)  # Adjust the number of columns as per your requirement
        num_cols = min(num_images, 3)

        # Create subplots with the specified number of rows and columns
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4))

        # Loop through the images and plot them
        for i, image in enumerate(preprocessed_data):
            row_idx = i // num_cols
            col_idx = i % num_cols

        # Plot it on the corresponding subplot
        axes[row_idx, col_idx].imshow(image)
        axes[row_idx, col_idx].axis('off')

        # Adjust the spacing between subplots
        fig.tight_layout()

        # Display the subplots
        plt.show()






    
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