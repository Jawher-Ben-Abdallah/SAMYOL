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
        self.images = images
        self.predictions = predictions

    def display(self, index) -> None:
        """
        Display the bounding boxes and masks.
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

    def save(self, save_dir="./", filename="default", fuse_masks=False, image_id=0, format="png"):
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
