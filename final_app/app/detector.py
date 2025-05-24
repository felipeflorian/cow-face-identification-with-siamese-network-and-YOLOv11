from PIL import Image
from typing import List, Dict, Any
from ultralytics import YOLO
import numpy as np
from PIL import Image
from pathlib import Path

#Ya no se usa
class DummyDetector:
    def predict(self, img: Image.Image) -> List[Dict[str, Any]]:
        w, h = img.size
        # bbox = [x, y, w, h]
        return [{"bbox": [int(w*0.25), int(h*0.25), int(w*0.5), int(h*0.5)],
                 "confidence": 0.99}]
    


class YoloDetector:
    def __init__(self, weights=None, conf=0.5, img_size=640):
        if weights is None:
            base_dir = Path(__file__).resolve().parent.parent
            weights = base_dir / "models" / "best_cow_recognizer.pt"

        self.model = YOLO(weights)
        self.conf  = conf
        self.imgsz = img_size

    def predict(self, img: Image.Image):
    
        frame = np.asarray(img)[..., ::-1]

        results = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            verbose=False
        )

        dets = []
        if results and results[0].boxes:
            boxes = results[0].boxes.xywh.cpu().numpy()      # x_c, y_c, w, h
            confs = results[0].boxes.conf.cpu().numpy()

            for (xc, yc, w, h), c in zip(boxes, confs):
                x0 = int(xc - w / 2)
                y0 = int(yc - h / 2)
                dets.append(
                    {"bbox": [x0, y0, int(w), int(h)],
                     "confidence": float(c)}
                )
        return dets
