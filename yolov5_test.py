import numpy as np
from PIL import Image
from SAMYOL import SAMYOL

def display_segmentation_masks(masks_path):
    masks = np.array(Image.open(masks_path))
    num_masks = masks.shape[-1]

    fig, axes = plt.subplots(1, num_masks, figsize=(10, 10))

    for i in range(num_masks):
        mask = masks[..., i]

        axes[i].imshow(mask, cmap="gray")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


device = "cpu"
sam_model_type = "base"
sam_source = "HuggingFace"

class_labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
    'hair drier', 'toothbrush']
    
input_paths = ["./assets/images/image1.jpg", "./assets/images/image2.jpg"]

if __name__ == "__main__":
    # Specify the YOLO model version and path
    version = "5"
    model_type= "yolov5s"

    samyol = SAMYOL(
        yolo_model_path=model_type,
        yolo_version=version,
        sam_model_type=sam_model_type,
        sam_source=sam_source,
        class_labels=class_labels,
        device=device,
    )

    # Generate predictions using YOLOv6 model + SAM 
    samyol_predictions = samyol.predict(input_paths=input_paths)
    # Dsiplay the original image with the bounding boxes and the corresponding masks
    img_idx = 0
    samyol_predictions.display(img_idx)