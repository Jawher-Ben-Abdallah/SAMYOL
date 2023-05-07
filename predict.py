import yolo_anything
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
        yolo_anything.download_model_weights("YOLO")
        args.yolo_checkpoint = "./checkpoints/yolov8s.pt"
    # if (not args.sam_checkpoint):
    #     yolo_anything.download_model_weights("SAM")
    #     args.sam_checkpoint = "./checkpoints/sam_vit_h_4b8939.pth"

    detections, list_of_masks = yolo_anything.predict(
        args.input,
        args.yolo_checkpoint,
        args.model_type,
        args.sam_checkpoint,
        args.device
    )
    