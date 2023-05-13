import cv2
import numpy as np
from PIL import Image
from utils import letterbox



class YOLOPostProcessing():
    def get_yolo_7_preprocessing(img, session): #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = img.copy()
        image, _, _ = letterbox(image, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)

        im = image.astype(np.float32)
        im /= 255

        outname = [i.name for i in session.get_outputs()]
        outname

        inname = [i.name for i in session.get_inputs()]
        inname

        inp = {inname[0]:im}

        return outname, inp




class YOLOPostProcessing():
    def __init__(self) -> None:
        print("Did a bunch of stuff")

    def get_yolo_6_postprocessing():
        print("Fetching yolo 6 postprocessing")
    
    def get_yolo_7_postprocessing(img, outputs, colors, names): #img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = img.copy()
        image, ratio, dwdh = letterbox(image, auto=False)

        ori_images = [img.copy()]

        for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(outputs):
            image = ori_images[int(batch_id)]
            box = np.array([x0,y0,x1,y1])
            box -= np.array(dwdh*2)
            box /= ratio
            box = box.round().astype(np.int32).tolist()
            cls_id = int(cls_id)
            score = round(float(score),3)
            name = names[cls_id]
            color = colors[name]
            name += ' '+str(score)
            cv2.rectangle(image,box[:2],box[2:],color,2)
            cv2.putText(image,name,(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2)  

        return Image.fromarray(ori_images[0]) #remove index
    
    def get_yolo_8_postprocessing(detections):
        object_detection_predictions = []
        for i, detection in enumerate(detections):
            class_labels = detection.names
            boxes = detection.boxes
            for box in boxes:
                object_detection_predictions.append(
                    {
                        'image_id': i,
                        'class_id': class_labels[int(box.cls.item())],
                        'bbox': box.xyxy[0].numpy().round().astype(np.int32).tolist()
                    }
                )
        return object_detection_predictions

    def get_yolo_nas_postprocessing():
        print("Fetching yolo nas postprocessing")