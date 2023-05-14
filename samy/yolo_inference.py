from ultralytics import YOLO
import onnxruntime as ort
from .utils import generic_ort_inference


class YOLOInference():

    @staticmethod
    def get_yolo_6_inference(model_path, inputs):
        detections = generic_ort_inference(model_path, inputs[0])
        resize_data, origin_RGB = inputs[1:]
        return detections, resize_data, origin_RGB


    def get_yolo_7_inference(weights_path, cuda, inp, outname):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        session = ort.InferenceSession(weights_path, providers=providers)
        return session.run(outname, inp)[0]
        

    def get_yolo_8_inference(weights_path, image):
        model = YOLO(weights_path)
        detections = model.predict(
            image, 
            conf=0.35, 
            iou=0.65, 
            verbose=False
            )
        return detections

    
    @staticmethod
    def get_yolo_nas_inference(inputs, model_type, model_path, classes):
        
        try:
            from super_gradients.training import models
        except ImportError:
            raise ImportError("super-gradients package required.")
        
        model = models.get(
            model_type,
            num_classes=len(classes),
            checkpoint_path=model_path
        )
        detections = model.predict(inputs)
        return detections
    

