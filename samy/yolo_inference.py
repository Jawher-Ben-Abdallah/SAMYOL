import onnxruntime as ort
from super_gradients.training import models


class YOLOInference():
    def __init__(self) -> None:
        pass

    def run_yolo_6_inference(inputs, model_path):
        providers = ["CPUExecutionProvider"]
        session = ort.InferenceSession(model_path, providers=providers)
        outname = [i.name for i in session.get_outputs()]
        inname = [i.name for i in session.get_inputs()]
        detections = session.run(outname,{inname[0]:inputs})
        return detections

    def run_yolo_7_inference():
        pass

    def run_yolo_8_inference():
        pass

    def run_yolo_nas_inference(inputs, model_type, model_path, classes):
        model = models.get(
            model_type,
            num_classes=len(classes),
            checkpoint_path=model_path
        )
        detections = model.predict(inputs)
        return detections