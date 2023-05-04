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
    "--image", type=str, required=True,
    help="The path to the input image."
)


if __name__ == "__main__":
    args = parser.parse_args()
    
    if (not args.yolo_checkpoint):
        utils.download_model_weights("YOLO")
    if (not args.sam_checkpoint):
        utils.download_model_weights("SAM")

    # run YOLO to get scene BBoxes
    # input = image (numpy array), output = list of dicts [{image_id: _, class_id: _, bbox: []}]

    # run SAM to get the segmentations
    # input = image and the list of dicts, output = segmentation masks (in any format Rim likes)

    




    