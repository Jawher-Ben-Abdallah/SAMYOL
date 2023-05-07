import numpy as np
from segment_anything import SamPredictor, sam_model_registry



def create_segment_anything_predictor (model_type, sam_checkpoint, device):
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor


def extract_bbox(list_of_dicts):
    return [item['bbox'] for item in list_of_dicts]


def generate_masks(image, input_bbox_list, predictor):
    predictor.set_image(image)
    masks_list = [predictor.predict(point_coords=None,
                                    point_labels=None,
                                    box=np.array(input_box)[None, :],
                                    multimask_output=False)[0]
                  for input_box in input_bbox_list]
    return masks_list



def segment_objects_in_image (model_type, sam_checkpoint, device, list_of_dicts, image):
    predictor = create_segment_anything_predictor (model_type, sam_checkpoint, device)
    list_of_bbox = extract_bbox(list_of_dicts)
    list_of_masks = generate_masks(image, list_of_bbox, predictor)
    return list_of_masks
     

