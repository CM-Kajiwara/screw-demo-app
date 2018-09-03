import sagemaker
import os
from sagemaker.predictor import RealTimePredictor

class ScrewPredictor(sagemaker.predictor.RealTimePredictor):
     
    def __init__(self, *args, **kwargs):
        from sagemaker.predictor import json_deserializer
        super().__init__(*args,  **kwargs)
        # screw_classifier_end = "screw-classifie"
        screw_classifier_end = os.getenv('SCREW_CLASSIFER_ENDPOINT',"screw-classifie")
        self._screw_classifier = sagemaker.predictor.RealTimePredictor(screw_classifier_end, content_type='image/jpeg', deserializer=json_deserializer)
        self._object_classes = ['screw']
        self._screw_classes = ['nabeneji','saraneji', 'tyouneji']


    @property
    def screw_classifier(self):
        """画像分類器"""
        return self._screw_classifier

    @property
    def object_classes(self):
        """物体検出時のラベル"""
        return self._object_classes

    @property
    def screw_classes(self):
        """画像分類時のラベル(ネジの種類)"""
        return self._screw_classes
         
    def predict(self, image_byte_array, thresh=0.6):
        from PIL import Image
        from io import BytesIO 
        pilImg = Image.open(BytesIO(image_byte_array))
        size = pilImg.size
        width = size[0]
        height = size[1]
        detections = super().predict(image_byte_array)
        results = []
        for i, det in enumerate(detections['prediction']):
            (klass, score, x0, y0, x1, y1) = det
            if score < thresh:
                continue
            cls_id = int(klass)
            xmin = int(x0 * width)
            ymin = int(y0 * height)
            xmax = int(x1 * width)
            ymax = int(y1 * height)
            if self._object_classes and len(self._object_classes) > cls_id:
               class_name = self._object_classes[cls_id]
               result = {
                   "class_name":class_name,
                   "xmin" : xmin,
                   "ymin" : ymin,
                   "xmax" : xmax,
                   "ymax" : ymax,
                   "score" : score
               } 
               # ネジの場合はネジの種類を分類
               if class_name == 'screw':
                   cropImg = pilImg.crop((xmin, ymin, xmax, ymax))
                   class_name, screw_prob = self.classify_screw(cropImg)
                   result["class_name"] = class_name
                   result["screw_prob"] = screw_prob
                   results.append(result)
        
        return results

    def classify_screw(self,pil_img):
        """
        ネジを分類します
        Parameters:
        ----------
        pil_img : PIL.Image
            screw image
        screw_classes : tuple or list of str
            class names
        """
        import io
        import numpy as np
        from PIL import Image
        import time
        
    
        pil_img = self.resize_img(pil_img, 224, 224)
        # pil_img.save("classify_img/"+ str(time.time()) + ".jpg")
        
        # byte形式に変換
        img_byte = io.BytesIO()
        pil_img.save(img_byte, format='JPEG')
        img_byte = img_byte.getvalue()
        
        # 種類判別をするために画像分類器に投げる
        response = self._screw_classifier.predict(img_byte)
        
        # 確率が一番高いものをその種類とする
        screw_id = np.argmax(response)
        screw_name = str(screw_id)
        if screw_id < len(self._screw_classes):
            screw_name = self._screw_classes[screw_id]
        return screw_name, response[screw_id] 

    def resize_img(self,pil_img, width, height):
        """
        画像サイズを変更します。足りない領域は黒塗りされます。
 
        Parameters:
        ----------
        pil_img : PIL.Image
            screw image
        width : int
        height : int
        """
        pil_img.thumbnail((width, height))
        pil_img = pil_img.crop((0, 0, width, height))
        return pil_img
         
