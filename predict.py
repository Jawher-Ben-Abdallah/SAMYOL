import utils
import argparse


parser = argparse.ArgumentParser(
    description="Perform Instance Segmentation with YOLO and SAM"
)

parser.add_argument(
    "--yolo-checkpoint", type=str, required=False,
    help="The path to the YOLO model checkpoint"
)

parser.add_argument(
    "--sam-checkpoint", type=str, required=False,
    help="The path to the SAM model checkpoint"
)

parser.add_argument(
    "--model-type", type=str, required=False,
    help="Model type"
)

parser.add_argument(
    "--input", type=str, required=True,
    help="The path to the input image."
)

parser.add_argument(
    "--device", type=str, default="cpu", required=False,
    help="Device on which to run the model."
)


if __name__ == "__main__":
    args = parser.parse_args()
    
    if (not args.yolo_checkpoint):
        utils.download_model_weights("YOLO")
    if (not args.sam_checkpoint):
        utils.download_model_weights("SAM")

    image = utils.load_image(args.input)

    # run YOLO to get scene BBoxes
    detections = {}
    # input = image (numpy array), output = list of dicts [{image_id: _, class_id: _, bbox: []}]

    # run SAM to get the segmentations
    list_of_masks = utils.segment_objects_in_image (args.model_type, 
                                                    args.sam_checkpoint, 
                                                    args.device, 
                                                    detections, 
                                                    image)
    

    




    