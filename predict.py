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
    "--input", type=str, required=True,
    help="The path to the input image."
)


if __name__ == "__main__":
    args = parser.parse_args()
    
    if (not args.yolo_checkpoint):
        utils.download_model_weights("YOLO")
        args.yolo_checkpoint = "./checkpoints/yolov8s.pt"
    # if (not args.sam_checkpoint):
    #     utils.download_model_weights("SAM")
    #     args.sam_checkpoint = "./checkpoints/sam_vit_h_4b8939.pth"

    image = utils.load_image(args.input)

    # run YOLO to get scene BBoxes
    detections = utils.get_postprocessed_detections(
        args.yolo_checkpoint,
        image
        )

    # run SAM to get the segmentations
    masks = {}
    # input = image and the list of dicts, output = segmentation masks (in any format Rim likes)

    




    