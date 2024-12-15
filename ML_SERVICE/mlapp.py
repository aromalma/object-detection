
from flask import Flask, request, Response,  render_template_string, jsonify
from ultralytics import YOLO
import threading
import numpy as np
import cv2
import os
from opencv_draw_annotation import draw_bounding_box
from PIL import Image
import io
import base64
import torch

app = Flask(__name__)

class detection:
    def __init__(self,):
        # check for cuda availablilty
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.model = YOLO('yolo11n.pt')

        self.model.to(self.device)
    def detect(self,input):
        return self.model(input)

obj = detection()
lock=threading.Lock()

@app.route('/',methods=["POST"]) 
def detectimg():
    
    if 'image' not in request.files:
        return "No image", 405
    try:
        img = request.files['image']
        img = np.frombuffer(img.read(), np.uint8)
        
        
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        names = obj.model.names

        # only one thread to access at a time
        with lock :
            result = []
            res = obj.detect(img.copy())
            print(res)
            # move to cpu before conveting to numpy array
            res_box = res[0].boxes.cpu().numpy()
            iterate = zip(res_box.cls.astype(int),res_box.conf.astype(float),res_box.xyxy.astype(int))
            # draw bbounding box and create json
            for cls_name ,confidence, bbox in iterate:
                b = bbox.tolist()
                result.append({"class":names[cls_name],"conf":confidence,"bxyxy":b})
                draw_bounding_box(img, b, labels=[names[cls_name]],
                    color='green')
            # prepare to send response back
            img = Image.fromarray(img.astype("uint8"))
            rawBytes = io.BytesIO()
            img.save(rawBytes, "JPEG")
            rawBytes.seek(0)
            img_base64 = base64.b64encode(rawBytes.read()).decode('utf-8')


        
            
        return jsonify({"image":img_base64,"json":result}),200
    except Exception as e:
        return jsonify({"error": "Error in detection", "details": str(e)}), 500



if __name__ == "__main__":
    app.run(host="0.0.0.0",port =8011,debug=True,)

# curl -i -X POST -H "Content-Type: multipart/form-data" -F "image=@/home/tl028/Downloads/201_265_219_1387.83_29_4824.jpeg" http://localhost:8011
