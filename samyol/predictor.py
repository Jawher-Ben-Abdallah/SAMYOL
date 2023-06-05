from samyol.yolo_preprocessing import YOLOPreProcessing
from samyol.yolo_inference import YOLOInference
from samyol.yolo_postprocessing import YOLOPostProcessing
from typing import Union, List, Optional, Dict, Tuple, Callable
from samyol.sam_inference import HuggingFaceSAMModel
import matplotlib.pyplot as plt
import numpy as np
import random

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
        object_segmentation_predictions = HuggingFaceSAMModel(input_paths, obj_det_predictions).sam_inference(self.device)
        return preprocessed_data[2], object_segmentation_predictions
    

    def display(self) -> None:
        """
        Display the bounding boxes and masks.
        """
        original_RGB, object_segmentation_predictions = self.predict()
        num_images = len(original_RGB)

        # Define the number of rows and columns for the subplots
        num_rows = int(num_images / 3) + (num_images % 3 > 0)  # Adjust the number of columns as per your requirement
        num_cols = min(num_images, 3)

        # Create subplots with the specified number of rows and columns
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4))


        # Loop through the data and plot each dictionary
        for i, d in enumerate(original_RGB):
            row_idx = i // num_cols
            col_idx = i % num_cols

            image = original_RGB[d['image_id']]  

            # Plot the image on the corresponding subplot
            axes[row_idx, col_idx].imshow(image)
            axes[row_idx, col_idx].axis('off')

            # Plot the bounding boxes
            for bbox, class_id in zip(d['bbox'], d['class_id']):
                x1, y1, x2, y2 = bbox
                color = random.random(), random.random(), random.random()  # Generate a random color for each class_id
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2)
                axes[row_idx, col_idx].add_patch(rect)

            # Plot the masks with low opacity
            for mask, class_id in zip(d['masks'], d['class_id']):
                color = random.random(), random.random(), random.random()  # Generate a random color for each class_id
                alpha = 0.4  # Set the opacity of the mask
                masked_image = np.where(mask[:, :, np.newaxis], image * (1 - alpha) + color + alpha, image)
                axes[row_idx, col_idx].imshow(masked_image)

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