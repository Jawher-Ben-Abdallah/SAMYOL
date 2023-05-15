from samy.yolo_preprocessing import YOLOPreProcessing
from samy.yolo_inference import YOLOInference
from samy.yolo_postprocessing import YOLOPostProcessing


class SAMY():
    def __init__(self):
        pass

    def predict(self, inputs, version, model_path):
        yolo_pipeline = self.get_yolo_pipeline(version)
        preprocessed_data = yolo_pipeline['preprocessing'](inputs)
        outputs = yolo_pipeline['inference'](model_path, preprocessed_data)
        obj_det_predictions = yolo_pipeline['postprocessing'](outputs)
        return obj_det_predictions

    @staticmethod
    def get_yolo_pipeline(version):
        run_yolo_preprocessing = getattr(YOLOPreProcessing, f"get_yolo_{version}_preprocessing")
        run_yolo_inference = getattr(YOLOInference, f"get_yolo_{version}_inference")
        run_yolo_postprocessing = getattr(YOLOPostProcessing, f"get_yolo_{version}_postprocessing")
        return {
            'preprocessing': run_yolo_preprocessing, 
            'inference': run_yolo_inference, 
            'postprocessing': run_yolo_postprocessing
            }