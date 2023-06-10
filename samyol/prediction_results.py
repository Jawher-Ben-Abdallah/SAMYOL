from typing import List, Dict
import torch
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import json


class SAMYOLPredictions():
    def __init__(
            self,
            images: List[np.ndarray],
            predictions: List[Dict]
    ):
        """
        Initialize SAMYOLPredictions.

        Args:
            images (List[np.ndarray]): List of input images.
            predictions (List[Dict]): List of prediction dictionaries.
        """
        self.images = images
        self.predictions = predictions

    def display(self, index: int) -> None:
        """
        Display the bounding boxes and masks for a specific image.

        Args:
            index (int): Index of the image to display.
        """
        image = self.images[index]
        target_predictions = self.predictions[index]

        # Create a subplot for displaying the image, bounding boxes, and masks
        fig, ax = plt.subplots(figsize=(6, 6))

        # Plot the image
        ax.imshow(image)

        # Plot the masks with low opacity
        colors = []
        for mask, class_id in zip(target_predictions['masks'], target_predictions['class_id']):
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
            h, w = mask.shape[-2:]
            bbox_mask = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            ax.imshow(bbox_mask)

        # Plot the bounding boxes
        for bbox, class_label in zip(target_predictions['bbox'], target_predictions['class_label']):
            x1, y1, x2, y2 = bbox
            color = (random.random(), random.random(), random.random()) # Generate a random color for each class_label
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2)
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, class_label, fontsize=8, color="black", weight="bold", bbox=dict(facecolor=color, edgecolor="black", alpha=0.8, pad=0.8))


        # Configure the plot
        ax.axis('off')
        plt.tight_layout()

        # Show the plot
        plt.show()

    def save(self, 
             save_dir: str="./", 
             filename: str="default", 
             fuse_masks: bool=False, 
             save_metadata: bool=False, 
             image_id:int=0, 
             mask_format: str="png") -> None:

        """
        Save the masks as images and metadata as JSON file.

        Args:
            save_dir (str): Directory to save the images.
            filename (str): Filename for the saved images.
            fuse_masks (bool): Whether to merge all masks or save them separately.
            save_metadata (bool): Whether to save metadata as a JSON file.
            image_id (int): Index of the image to save.
            mask_format (str): Image format to save.
        """

        filtered_data = self.predictions[image_id]
        masks = filtered_data['masks']
        if fuse_masks:
            # Merge all masks
            masks = np.logical_or.reduce(np.array(masks))
            cv2.imwrite(f"{save_dir}/{filename}.{mask_format}", masks * 255)
        else:
            # Per Mask
            class_ids = self.predictions[image_id]['class_id']
            masks = [mask * (class_id + 1) for (mask, class_id) in zip (masks, class_ids)]
            masks = np.stack(masks, axis=2)
            masks = masks.astype(np.uint8)
            cv2.imwrite(f"{save_dir}/{filename}.{mask_format}", masks)
        
        if save_metadata:
            metadata = {
                'image_id': image_id, 
                'class_id': filtered_data['class_id'], 
                'class_label':filtered_data['class_label'],
                'score':  filtered_data['score'], 
                'bbox': filtered_data['bbox'],
                }
            # Save dictionary as JSON file
            with open(f"{save_dir}/{filename}.json", "w", encoding="utf-8") as json_file:
                json.dump(metadata, json_file, indent=4)
