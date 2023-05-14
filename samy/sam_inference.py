from PIL import Image
from transformers import SamModel, SamProcessor


##############################################################

# Transformers installation
#! pip install transformers datasets
# To install from source instead of the last release, comment the command above and uncomment the following one.
# ! pip install git+https://github.com/huggingface/transformers.git

##############################################################




class HuggingFaceSAMModel :
    def __init__ (self, image_path, bbox):
        self.bbox = bbox
        self.model, self.processor = self.load_model()
        self.image = Image.open(image_path).convert("RGB")
        

    def load_model (self):
        sam_model = SamModel.from_pretrained("facebook/sam-vit-huge")
        sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
        return sam_model, sam_processor

    def sam_inference(self, device):
        inputs = self.processor(self.image,  input_boxes=[[self.bbox]], return_tensors="pt").to(device)
        outputs = self.model(**inputs)
        masks = self.processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
        scores = outputs.iou_scores
        return masks, scores