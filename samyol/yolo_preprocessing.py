from .utils import generic_yolo_preprocessing


class YOLOPreProcessing():
    
    @staticmethod
    def get_yolo_6_preprocessing(inputs):
        return generic_yolo_preprocessing(inputs)

    @staticmethod
    def get_yolo_7_preprocessing(inputs): 
        return generic_yolo_preprocessing(inputs)

    @staticmethod
    def get_yolo_8_preprocessing(inputs):
        return inputs
    
    @staticmethod
    def get_yolo_nas_preprocessing(inputs):
        return inputs