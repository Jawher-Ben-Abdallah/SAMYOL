from .utils import check_and_install_library
from typing import List, Dict
import numpy as np
import torch


class HuggingFaceSAMModel :
    def __init__ (self, 
                  original_RGB: List[np.ndarray], 
                  obj_det_predictions: List[Dict], 
                  device: str
                  ):
        """
        Initialize the HuggingFaceSAMModel.

        Args:
            original_RGB (List[np.ndarray]): List of RGB images.
            obj_det_predictions (List[Dict]): Object detection predictions.
            device (str): The device to run the inference on.
        """
        self.device = device
        self.original_RGB = original_RGB
        self.obj_det_predictions = obj_det_predictions
        self.model, self.processor = self.load_model()

    def load_model (self):
        """
        Load the HuggingFace SAM model and processor.

        Returns:
            Tuple: A tuple containing the loaded SAM model and processor.
        """
        check_and_install_library('transformers')
        from transformers import SamModel, SamProcessor

        sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(self.device)
        sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        return sam_model, sam_processor
    

    def sam_inference(self) -> List[Dict]:
        """
        Perform inference using the HuggingFace SAM model.

        Args:
            device (str): The device to run the inference on.

        Returns:
            List[Dict]: Object segmentation predictions as a list of dictionaries.
        """

        object_segmentation_predictions = []

        image_ids = list(set([d['image_id'] for d in self.obj_det_predictions]))
    
        for image_id in image_ids:
            # Filter the data based on the current image_id
            filtered_data = [d for d in self.obj_det_predictions if d['image_id'] == image_id]

            # Extract the bounding boxes for the current image_id
            bboxes = [d['bbox'] for d in filtered_data]

            # Extract the class ids and class labels for the current image_id
            class_ids = [d['class_id'] for d in filtered_data]
            class_labels = [d['class_label'] for d in filtered_data]
            
            # Get the image based on the current image_id
            image = self.original_RGB[image_id]

            # Perform the inference for the current image_id
            inputs = self.processor(image, input_boxes=[[bboxes]], return_tensors="pt").to(self.device)
            image_embeddings = self.model.get_image_embeddings(inputs["pixel_values"])
            inputs.pop("pixel_values", None)
            inputs.update({"image_embeddings": image_embeddings})
            with torch.no_grad():
                outputs = self.model(**inputs)
            masks = self.processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
            scores = outputs.iou_scores

            # Reshape the tensor to have size (N, 3)
            reshaped_iou_scores = outputs.iou_scores.squeeze()
            idx_max_iou = torch.argmax(reshaped_iou_scores.view(-1, 3), dim=1).tolist()
              
            object_segmentation_predictions.append({
                    'image_id': image_id,
                    'class_id': class_ids,
                    'class_label': class_labels,
                    'score': [outputs.iou_scores.squeeze()[i, j, ...].item() for i, j in enumerate (idx_max_iou)],
                    'bbox': bboxes,
                    'masks': [masks[0][i, j, ...] for i, j in enumerate (idx_max_iou)]
                })

        return object_segmentation_predictions