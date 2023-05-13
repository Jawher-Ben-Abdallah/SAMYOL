from ultralytics import YOLO
from utils import letterbox
import onnxruntime as ort
from super_gradients.training import models


class YOLOInference():
    def __init__(self) -> None:
        pass

    def run_yolo_6_inference():
        pass

    def run_yolo_7_inference(weights_path, cuda, inp, outname):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        session = ort.InferenceSession(weights_path, providers=providers)
        return session.run(outname, inp)[0]
        


    def run_yolo_8_inference(weights_path, image):
        model = YOLO(weights_path)
        detections = model.predict(
            image, 
            conf=0.35, 
            iou=0.65, 
            verbose=False
            )
        return detections

    def run_yolo_nas_inference(inputs, model_type, model_path, classes):
        model = models.get(
            model_type,
            num_classes=len(classes),
            checkpoint_path=model_path
        )
        return model.predict(inputs)