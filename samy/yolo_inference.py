from super_gradients.training import models


class YOLOInference():
    def __init__(self) -> None:
        pass

    def run_yolo_6_inference():
        pass

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
        return model.predict(inputs)