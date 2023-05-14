from utils import load_image
from sam_inference import HuggingFaceSAMModel
from yolo_postprocessing import YOLOPostProcessing
from yolo_inference import YOLOInference
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np



def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    

image_path = 'HRPlanes/test/000_set2.jpg'
image = load_image(image_path)
weights_path = 'checkpoints/yolov8s.pt'
# device = "cuda" if torch.cuda.is_available() else "cpu"













if __name__ =='__main__':

  detections = YOLOInference().run_yolo_8_inference(weights_path, image)
  print(detections)
  yolov8_bboxes = YOLOPostProcessing().get_yolo_8_postprocessing(detections)
  bbox = yolov8_bboxes[0]['bbox']
  HF_sam = HuggingFaceSAMModel (image_path, bbox)
  sam_model = HF_sam.load_model()
  masks, scores = HF_sam.sam_inference("cpu")
  print(masks.shape)
  print(scores.shape)

  print('Joujou ya passive aggressive')


# plt.imshow(np.array(raw_image))
# ax = plt.gca()
# for mask in outputs["masks"]:
#     show_mask(mask, ax=ax, random_color=True)
# plt.axis("off")
# plt.show()