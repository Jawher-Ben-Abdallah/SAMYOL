from typing import List, Dict
import numpy as np

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

    def display(self):
        ...

    def save(self):
        ...
