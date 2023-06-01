import subprocess
from PIL import Image
from typing import List, Tuple
import numpy as np

class HuggingFaceSAMModel :
    def __init__ (self, image_path: str, bbox: List[float]): 

        """
        Initialize the HuggingFaceSAMModel.

        Args:
            image_path (str): Path to the image file.
            bbox (List[float]): The bounding box coordinates as a list of floats [x1, y1, x2, y2].
        """
        self.bbox = bbox
        self.model, self.processor = self.load_model()
        self.image = Image.open(image_path).convert("RGB")
        

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
        inputs = self.processor(self.image,  input_boxes=[[self.bbox]], return_tensors="pt").to(device)
        outputs = self.model(**inputs)
        masks = self.processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
        scores = outputs.iou_scores
        return masks, scores