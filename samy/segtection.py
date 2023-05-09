#######################################
## ALL THE COOL NAMES WERE TAKEN!!!! ##
#######################################

import utils

def predict(
        input,
        yolo_checkpoint,
        sam_model_type,
        sam_checkpoint,
        device
):
    image = utils.load_image(input)

    # run YOLO to get scene BBoxes
    detections = utils.get_postprocessed_detections(
        yolo_checkpoint,
        image
        )

    # run SAM to get the segmentations
    list_of_masks = utils.segment_objects_in_image (
        sam_model_type, 
        sam_checkpoint, 
        device, 
        detections, 
        image
        ) 

    return detections, list_of_masks