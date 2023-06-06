# from utils import load_image
# from sam_inference import HuggingFaceSAMModel
# from yolo_postprocessing import YOLOPostProcessing
# from yolo_inference import YOLOInference
# import matplotlib.pyplot as plt
# from PIL import Image
# import numpy as np

from samyol.predictor import SAMYOL


input_paths = "./assets/0_DT7.jpg"
model_path = "./checkpoints/yolov8s.pt"
version = "8"
device = "cpu"



if __name__ =='__main__':

    samyol = SAMYOL(
        model_path=model_path,
        version=version,
        device=device,
    )

    predictions = samyol.predict(
        input_paths=input_paths,
    )

    samyol.display()


    