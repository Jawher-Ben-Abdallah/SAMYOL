from typing import List, Dict
import torch
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt


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
        ax.axis('off')

        # Plot the bounding boxes
        for bbox, class_id in zip(target_predictions['bbox'], target_predictions['class_id']):
            x1, y1, x2, y2 = bbox
            color = random.random(), random.random(), random.random()  # Generate a random color for each class_id
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2)
            ax.add_patch(rect)

        # Plot the masks with low opacity
        for mask, class_id in zip(target_predictions['masks'], target_predictions['class_id']):
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
            h, w = mask.shape[-2:]
            bbox_mask = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            ax.imshow(bbox_mask)

        plt.show()

    def save(self, save_dir: str="./", filename: str="default", fuse_masks: bool=False, image_id:int=0, format: str="png") -> None:
        """
        Save the masks as images.

        Args:
            save_dir (str): Directory to save the images.
            filename (str): Filename for the saved images.
            fuse_masks (bool): Whether to merge all masks or save them separately.
            image_id (int): Index of the image to save.
            format (str): Image format to save.
        """
        masks = self.predictions[image_id]['masks']
        if fuse_masks:
            # Merge all masks
            masks = [mask.numpy() for mask in masks]
            masks = np.logical_or.reduce(np.array(masks))
            cv2.imwrite(f"{save_dir}/{filename}.{format}", masks * 255)
        else:
            # Per Mask
            class_ids = self.predictions[image_id]['class_id']
            masks = [mask * (class_id + 1) for (mask, class_id) in zip (masks, class_ids)]
            masks = torch.stack(masks, dim=-1)
            masks = masks.numpy().astype(np.uint8)
            cv2.imwrite(f"{save_dir}/{filename}.{format}", masks)
