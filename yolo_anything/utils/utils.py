import os
import cv2
from urllib.request import urlretrieve
from urllib.error import URLError
import numpy as np
import matplotlib.pyplot as plt

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


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 