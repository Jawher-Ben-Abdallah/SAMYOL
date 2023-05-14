from ultralytics import YOLO
import onnxruntime as ort
from super_gradients.training import models


class YOLOInference():

    def run_yolo_6_inference(self, model_path, inputs):
        detections = self.generic_ort_inference(model_path, inputs[0])
        resize_data, origin_RGB = inputs[1:]
        return detections, resize_data, origin_RGB

    def run_yolo_7_inference(self, weights_path, cuda, inp, outname):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        session = ort.InferenceSession(weights_path, providers=providers)
        return session.run(outname, inp)[0]
        

    def run_yolo_8_inference(self, weights_path, image):
        model = YOLO(weights_path)
        detections = model.predict(
            image, 
            conf=0.35, 
            iou=0.65, 
            verbose=False
            )
        return detections

    def run_yolo_nas_inference(self, inputs, model_type, model_path, classes):
        model = models.get(
            model_type,
            num_classes=len(classes),
            checkpoint_path=model_path
        )
        detections = model.predict(inputs)
        return detections
    
    @staticmethod
    def generic_ort_inference(model_path, inputs):
        providers = ["CPUExecutionProvider"]
        session = ort.InferenceSession(model_path, providers=providers)
        outname = [i.name for i in session.get_outputs()]
        inname = [i.name for i in session.get_inputs()]
        detections = session.run(outname,{inname[0]: inputs})
        return detections

