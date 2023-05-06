from ultralytics import YOLO


def run_object_detection(weights_path, image):
    model = YOLO(weights_path)
    results = model.predict(image)
    return results