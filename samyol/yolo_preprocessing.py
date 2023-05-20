import numpy as np
from .utils import letterbox, generic_yolo_preprocessing


class YOLOPreProcessing():
    
    @staticmethod
    def get_yolo_6_preprocessing(inputs):
        return generic_yolo_preprocessing(inputs)

    def get_yolo_7_preprocessing(inputs): 
        return generic_yolo_preprocessing(inputs)

    def get_yolo_8_preprocessing(inputs):
        return inputs