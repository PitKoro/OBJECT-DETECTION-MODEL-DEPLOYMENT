import io
import json
import base64
from typing import List, Dict

import kserve
from PIL import Image
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.structures.boxes import Boxes


class DetectorOutput:
    def __init__(self, pred_classes: torch.Tensor, pred_boxes: Boxes, scores: torch.Tensor, labels: List):
        self.pred_classes = [labels[id] for id in pred_classes.tolist()]
        self.pred_boxes = pred_boxes.tensor.tolist()
        self.scores = scores.tolist()


class MyModel(kserve.Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = False
        self.model = None
        self.gpu = False

    def get_labels_from_file(self, file_path=f"/app/configs/labels.txt") -> List:
        with open(file_path, "r") as labels_file:
            labels = labels_file.read().split('\n')
        return labels

    def load(self, config_path = "/app/configs/custom_faster_rcnn_R_50_FPN_3x.yaml") -> None:
        cfg = get_cfg()
        cfg.merge_from_file(config_path)
        self.labels = self.get_labels_from_file()
        self.model = DefaultPredictor(cfg)
        self.ready = True

    def predict(self, request_data: Dict, request_headers: Dict) -> Dict:
        img_base64 = request_data["image"]
        img = np.array(Image.open(io.BytesIO(base64.decodebytes(bytes(img_base64, "utf-8")))))
        result = self.model(img)
        outputs = DetectorOutput(result["instances"].pred_classes, result["instances"].pred_boxes, result["instances"].scores, self.labels)
        return {"predictions": json.loads(outputs.__dict__)}


if __name__ == "__main__":
    model = MyModel("detectron")
    model.load()
    kserve.ModelServer().start([model])