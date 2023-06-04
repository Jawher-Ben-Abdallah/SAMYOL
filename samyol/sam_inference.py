import subprocess
from PIL import Image
from typing import List, Tuple
import numpy as np

class HuggingFaceSAMModel :
    def __init__ (self, image_paths: str, obj_det_predictions: List[float]): 

        """
        Initialize the HuggingFaceSAMModel.

        Args:
            image_path (str): Path to the image file.
            bbox (List[float]): The bounding box coordinates as a list of floats [x1, y1, x2, y2].
        """
        self.image_paths = image_paths
        self.obj_det_predictions = obj_det_predictions
        self.model, self.processor = self.load_model()
        

    def load_model (self):
        """
        Load the HuggingFace SAM model and processor.

        Returns:
            Tuple: A tuple containing the loaded SAM model and processor.
        """
        try:
            from transformers import SamModel, SamProcessor
        except ImportError:
            print('Installing transformers ...')
            subprocess.check_call(["python", '-m', 'pip', 'install', 'transformers', 'datasets'])
        sam_model = SamModel.from_pretrained("facebook/sam-vit-huge")
        sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
        return sam_model, sam_processor

    def sam_inference(self, device: str) -> Tuple[List[np.ndarray], List[float]]:
        """
        Perform inference using the HuggingFace SAM model.

        Args:
            device (str): The device to run the inference on.

        Returns:
            Tuple[List[np.ndarray], List[float]]: A tuple containing the predicted masks (as a list of NumPy arrays) and the scores (as a list of floats).
        """
        object_segmentation_predictions = []

        self.image_ids = [d['image_id'] for d in self.obj_det_predictions]
    
        for image_id in self.image_ids:
            # Filter the data based on the current image_id
            filtered_data = [d for d in self.obj_det_predictions if d['image_id'] == image_id]

            # Extract the bounding boxes for the current image_id
            bboxes = [d['bbox'] for d in filtered_data]

            # Extract the bounding boxes for the current image_id
            class_ids = [d['class_id'] for d in filtered_data]
            
            # Load and preprocess the image based on the current image_id
            image = Image.open(self.image_paths[image_id]).convert("RGB")

            # Perform the inference for the current image_id
            inputs = self.processor(image,  input_boxes=[[bboxes]], return_tensors="pt").to(device)
            outputs = self.model(**inputs)
            masks = self.processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
            

            object_segmentation_predictions.append({
                    'image_id': image_id,
                    'class_id': class_ids,
                    'score': outputs.iou_scores,
                    'bbox': bboxes,
                    'masks': masks
                })

        return object_segmentation_predictions