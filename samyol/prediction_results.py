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
            predictions: List[Dict],
            class_labels: List[str]
    ):
        self.images = images
        self.predictions = predictions
        self.class_labels = class_labels

    def display(self) -> None:
        """
        Display the bounding boxes and masks.
        """
        num_images = len(self.images)
        
        # Define the number of rows and columns for the subplots
        num_rows = int(num_images / 3) + (num_images % 3 > 0)  # Adjust the number of columns as per your requirement
        num_cols = min(num_images, 3)

        # Create subplots with the specified number of rows and columns
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4), squeeze=False)


        # Loop through the data and plot each dictionary
        for i, d in enumerate(self.predictions):
            row_idx = i // num_cols
            col_idx = i % num_cols

            image = self.images[d['image_id']]  

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
                color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
                h, w = mask.shape[-2:]
                bbox_mask = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
                axes[row_idx, col_idx].imshow(bbox_mask)

        # Adjust the spacing between subplots
        fig.tight_layout()

        # Display the subplots
        plt.show()

    def save(self, save_dir, filename, image_id=0, format="jpg"):
        masks = self.predictions[image_id]['masks']
        masks = torch.logical_or(*masks)
        masks = masks.numpy().astype(np.uint8)
        cv2.imwrite(f"{save_dir}/{filename}.{format}", masks * 255)
