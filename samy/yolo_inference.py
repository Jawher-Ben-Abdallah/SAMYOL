from ultralytics import YOLO
from utils import letterbox
import onnxruntime as ort



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

    def run_yolo_nas_inference():
        pass