import subprocess
# import onnxruntime as ort
# from super_gradients.training import models


class YOLOInference():
    def __init__(self) -> None:
        pass

    def run_yolo_6_inference(self):
        pass

    def run_yolo_7_inference(self, weights_path, cuda, inp, outname):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        session = ort.InferenceSession(weights_path, providers=providers)
        return session.run(outname, inp)[0]
        


    def run_yolo_8_inference(self, weights_path, image):

        try:
            from ultralytics import YOLO
        except ImportError:
            print('Installing ultralytics ...')
            subprocess.check_call(["python", '-m', 'pip', 'install', 'ultralytics'])
                

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
        return model.predict(inputs)