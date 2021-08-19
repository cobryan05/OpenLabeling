import sys
import torch
import cv2
import numpy as np
import os
sys.path.append( os.path.join( os.path.dirname( __file__ ), "..", "submodules", "yolov5" ) )
from models.experimental import attempt_load
from utils.general import non_max_suppression

class YoloInference:
    def __init__(self, weights):
        self._yolo = attempt_load( weights )
        self._yolo.eval()
        self._device = torch.device('cpu')


    def runInference(self, img):
        yoloImg, ratio, (width,height) = letterbox( img, 640 )
        yoloImg = yoloImg[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB and HWC to CHW
        yoloImg = np.ascontiguousarray(yoloImg)
        yoloImg = torch.from_numpy(yoloImg).to(self._device)
        yoloImg = yoloImg.float()  # uint8 to fp16/32
        yoloImg /= 255.0  # 0 - 255 to 0.0 - 1.0
        if yoloImg.ndimension() == 3:
            yoloImg = yoloImg.unsqueeze(0)
        _,_,yoloImgH,yoloImgW = yoloImg.shape

        preds = self._yolo(yoloImg)[0]
        preds = non_max_suppression( preds )

        results = []
        for pred in preds:
            for obj in pred:
                x1,y1,x2,y2,conf,objclass = obj
                x1 -= width
                x2 -= width
                y1 -= height
                y2 -= height
                pX = ( ( x1 + x2 ) / 2 ) / yoloImgW
                pY = ( ( y1 + y2 ) / 2 ) / yoloImgH
                pW = ( abs(x2-x1)/yoloImgW )
                pH = ( abs(y2-y1)/yoloImgH )
                results.append( ( ( pX, pY ), (pW, pH ), conf, objclass ) )
        return results


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


