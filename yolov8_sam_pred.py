# from utils import load_image
# from sam_inference import HuggingFaceSAMModel
# from yolo_postprocessing import YOLOPostProcessing
# from yolo_inference import YOLOInference
# import matplotlib.pyplot as plt
# from PIL import Image
# import numpy as np

from samyol.predictor import SAMYOL
from samyol.prediction_results import SAMYOLPredictions


#input_paths = "./assets/0_DT7.jpg"
input_paths = ["./assets/image1.jpg", "./assets/image2.jpg", "./assets/dog.jpg"]
model_path = "./checkpoints/yolov8s.pt"
version = "8"
device = "cpu"
class_labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
    'hair drier', 'toothbrush']



if __name__ =='__main__':


    samyol = SAMYOL(
        model_path=model_path,
        version=version,
        device=device,
        class_labels=class_labels
    )

    # Generate predictions using YOLOv6 model + SAM 
    samyol_predictions = samyol.predict(input_paths=input_paths)
    samyol_predictions.display(0)