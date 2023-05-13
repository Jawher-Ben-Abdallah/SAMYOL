import numpy as np
from utils import letterbox



class YOLOPreProcessing():
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