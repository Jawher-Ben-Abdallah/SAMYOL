import os
import cv2
from urllib.request import urlretrieve
from urllib.error import URLError

def download_model_weights(model):
    
    root = "checkpoints"
    match model:
        case "YOLO":
            # download YOLOs by default
            url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt"
        case "SAM":
            # downlaod SAM: ViT-H by default
            url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        case _:
            raise ValueError("Invalid Model.")
        
    file_name = os.path.basename(url)
    os.makedirs(root, exist_ok=True)
    file_path = os.path.join(root, file_name)
    
    if not os.path.isfile(file_path):
        try:
            print(f"Downloading {model} weights to {file_path} from {url}")
            urlretrieve(url, file_path)
        except (URLError, IOError) as _:
            print(f"Could not download {model} Model Weights.")


def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
