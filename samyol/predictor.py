from samyol.yolo_preprocessing import YOLOPreProcessing
from samyol.yolo_inference import YOLOInference
from samyol.yolo_postprocessing import YOLOPostProcessing
from .utils import load_image


class SAMYOL():
    def __init__(self, input_paths, model_path, version):
        self.input_paths = input_paths
        self.model_path = model_path
        self.version = version

    def predict(self, extra_args=None):
        yolo_pipeline = self.get_yolo_pipeline(self.version)
        preprocessed_data = yolo_pipeline['preprocessing'](self.inputs)
        outputs = yolo_pipeline['inference'](self.model_path, preprocessed_data, **extra_args)
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
