import numpy as np
from utils.utils import load_image
from utils.yolo_utils import letterbox



class YOLOPreProcessing():
    
    def get_yolo_6_preprocessing(self, inputs):
        return self.generic_yolo_preprocessing(inputs)

    def get_yolo_7_preprocessing(img, session): #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = img.copy()
        image, _, _ = letterbox(image, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)

        im = image.astype(np.float32)
        im /= 255

        outname = [i.name for i in session.get_outputs()]
        inname = [i.name for i in session.get_inputs()]
        inp = {inname[0]:im}

        return outname, inp
    

    @staticmethod
    def generic_yolo_preprocessing(inputs):
        resize_data = []
        origin_RGB = []
        for image_path in inputs:
            image = load_image(image_path)
            origin_RGB.append(image)
            image, ratio, dwdh = letterbox(image)
            image = image.transpose((2, 0, 1))
            image = np.expand_dims(image, 0)
            image = np.ascontiguousarray(image)
            image = image.astype(np.float32)
            image /= 255
            resize_data.append((image, ratio, dwdh))
        np_batch = np.concatenate(data[0] for data in resize_data)
        return np_batch, resize_data, origin_RGB

    
