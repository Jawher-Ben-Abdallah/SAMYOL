import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple
import importlib
import subprocess


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from the given path and convert it to a NumPy array.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: The loaded image as a NumPy array.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def letterbox(
        im: np.ndarray, 
        new_shape: Tuple[int, int] =(640, 640), 
        color: Tuple[int, int, int]=(114, 114,  114), 
        auto: bool=True, 
        scaleup: bool=True, 
        stride: int=32
        ) -> Tuple[np.ndarray, float, Tuple[float, float]]:

    """
    Resize and pad the image while meeting stride-multiple constraints.

    Args:
        im (np.ndarray): Input image as a NumPy array.
        new_shape (Tuple[int, int]): Desired new shape of the image (height, width).
        color (Tuple[int, int, int]): RGB color value for the padding.
        auto (bool): Flag indicating whether to use the minimum rectangle for padding.
        scaleup (bool): Flag indicating whether to scale up the image if needed.
        stride (int): Stride value for meeting the stride-multiple constraints.

    Returns:
        Tuple[np.ndarray, float, Tuple[float, float]]: Resized and padded image, scale ratio, and padding values.
    """

    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)


def generic_yolo_preprocessing(
        inputs: List[str]
        ) -> Tuple[np.ndarray, List[Tuple[np.ndarray, float, Tuple[float, float]]], List[np.ndarray]]:
    """
    Perform generic YOLO preprocessing on a list of input images.

    Args:
        inputs (List[str]): List of input image paths.

    Returns:
        Tuple[np.ndarray, List[Tuple[np.ndarray, float, Tuple[float, float]]], List[np.ndarray]]: Preprocessed data, including the batch of resized images, ratio and padding information for each image, and the original RGB images.
    """
    resize_data = []
    origin_RGB = []
    for image_path in inputs:
        image = load_image(image_path)
        origin_RGB.append(image)
        image, ratio, dwdh = letterbox(image, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)
        image = image.astype(np.float32)
        image /= 255
        resize_data.append((image, ratio, dwdh))
    np_batch = np.concatenate([data[0] for data in resize_data])
    return np_batch, resize_data, origin_RGB


def generic_ort_inference(
        model_path: str, 
        inputs: np.ndarray, 
        cuda: bool = False
        ) -> np.ndarray:
    """
    Perform inference using the ONNX model with ONNX Runtime.

    Args:
        model_path (str): Path to the ONNX model file.
        inputs (np.ndarray): Input data for the model as a NumPy array.
        cuda (bool): Flag indicating whether to use CUDA for inference (default: True).

    Returns:
        np.ndarray: Inference results as a NumPy array.
    """
    providers = ['CUDAExecutionProvider'] if cuda else ['CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers)
    outname = [i.name for i in session.get_outputs()]
    inname = [i.name for i in session.get_inputs()]
    detections = session.run(outname,{inname[0]: inputs})
    return detections


def check_and_install_library(library_name):
    try:
        importlib.import_module(library_name)
        print(f"{library_name} is already installed.")
    except ImportError:
        print(f"{library_name} is not installed. Installing...")
        subprocess.check_call(['pip', 'install', library_name])
        print(f"{library_name} has been successfully installed.")