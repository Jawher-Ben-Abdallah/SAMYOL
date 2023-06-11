from .utils import check_and_install_library, download_model_weights
from typing import List, Dict
import numpy as np
import torch


class SAM:
    @staticmethod
    def predict_from_Meta(
        sam_model_type: str,
        original_RGB: List[np.ndarray], 
        obj_det_predictions: List[Dict], 
        device: str
    ) -> List[Dict]:
        """
        Predict object segmentation using SAM from Meta.

        Args:
            sam_model_type (str): SAM model type to use: base, large or huge.
            original_RGB (List[np.ndarray]): List of RGB images.
            obj_det_predictions (List[Dict]): Object detection predictions.
            device (str): Device for inference.

        Returns:
            List[Dict]: List of object segmentation predictions.
        """
        # Load Model
        check_and_install_library('segment_anything')
        from segment_anything import SamPredictor, sam_model_registry
        model_type, model_path = download_model_weights(sam_model_type)
        sam = sam_model_registry[model_type](checkpoint=model_path).to(device)
        predictor = SamPredictor(sam)

        # Get predictions from SAM
        object_segmentation_predictions = []

        image_ids = list(set([d['image_id'] for d in obj_det_predictions]))

        for image_id in image_ids:
            # Filter the data based on the current image_id
            filtered_data = [d for d in obj_det_predictions if d['image_id'] == image_id]

            # Extract the bounding boxes for the current image_id
            bboxes = [d['bbox'] for d in filtered_data]

            # Extract the class ids and class labels for the current image_id
            class_ids = [d['class_id'] for d in filtered_data]
            class_labels = [d['class_label'] for d in filtered_data]

            # Get the image based on the current image_id
            image = original_RGB[image_id]

            # Transform input BBoxes
            input_boxes = torch.tensor(
                bboxes, 
                device=predictor.device
            )

            predictor.set_image(image)
            transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
            masks, scores, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=True
            )
            idx_max_iou = torch.argmax(scores.view(-1, 3), dim=1).tolist()

            object_segmentation_predictions.append({
                'image_id': image_id,
                'class_id': class_ids,
                'class_label': class_labels,
                'score': [scores.squeeze()[i, j, ...].item() for i, j in enumerate(idx_max_iou)],
                'bbox': bboxes,
                'masks': [masks[i, j, ...].cpu().numpy() for i, j in enumerate(idx_max_iou)]
            })

        return object_segmentation_predictions

    @staticmethod
    def predict_from_HuggingFace(
        sam_model_type: str,
        original_RGB: List[np.ndarray], 
        obj_det_predictions: List[Dict], 
        device: str
    ) -> List[Dict]:
        """
        Predict object segmentation using SAM from Meta.

        Args:
            sam_model_type (str): SAM model type to use: base, large or huge.
            original_RGB (List[np.ndarray]): List of RGB images.
            obj_det_predictions (List[Dict]): Object detection predictions.
            device (str): Device for inference.

        Returns:
            List[Dict]: List of object segmentation predictions.
        """
        # Load SAM
        check_and_install_library('transformers')
        from transformers import SamModel, SamProcessor

        sam_model = SamModel.from_pretrained(f"facebook/sam-vit-{sam_model_type}").to(device)
        sam_processor = SamProcessor.from_pretrained(f"facebook/sam-vit-{sam_model_type}")

        # Get predictions from SAM
        object_segmentation_predictions = []

        image_ids = list(set([d['image_id'] for d in obj_det_predictions]))

        for image_id in image_ids:
            # Filter the data based on the current image_id
            filtered_data = [d for d in obj_det_predictions if d['image_id'] == image_id]

            # Extract the bounding boxes for the current image_id
            bboxes = [d['bbox'] for d in filtered_data]

            # Extract the class ids and class labels for the current image_id
            class_ids = [d['class_id'] for d in filtered_data]
            class_labels = [d['class_label'] for d in filtered_data]

            # Get the image based on the current image_id
            image = original_RGB[image_id]

            # Perform the inference for the current image_id
            inputs = sam_processor(image, input_boxes=[[bboxes]], return_tensors="pt").to(device)
            image_embeddings = sam_model.get_image_embeddings(inputs["pixel_values"])
            inputs.pop("pixel_values", None)
            inputs.update({"image_embeddings": image_embeddings})
            with torch.no_grad():
                outputs = sam_model(**inputs)
            masks = sam_processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
            scores = outputs.iou_scores

            # Reshape the tensor to have size (N, 3)
            reshaped_iou_scores = outputs.iou_scores.squeeze()
            idx_max_iou = torch.argmax(reshaped_iou_scores.view(-1, 3), dim=1).tolist()

            object_segmentation_predictions.append({
                'image_id': image_id,
                'class_id': class_ids,
                'class_label': class_labels,
                'score': [outputs.iou_scores.squeeze()[i, j, ...].item() for i, j in enumerate(idx_max_iou)],
                'bbox': bboxes,
                'masks': [masks[0][i, j, ...].cpu().numpy() for i, j in enumerate(idx_max_iou)]
            })

        return object_segmentation_predictions