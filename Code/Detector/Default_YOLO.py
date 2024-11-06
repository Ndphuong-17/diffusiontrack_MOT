from ultralytics import YOLO, ASSETS
from ultralytics.engine.results import Results
from ultralytics.utils.ops import scale_boxes
import cv2
from types import MethodType

from Code.Detector.Get_feature_yolo import _predict_once, non_max_suppression, get_object_features


class Default_YOLO:
    def __init__(self, path_yolo):
        self.model = self.init_yolo(path_yolo)
    
    def init_yolo(self, path_yolo):
        model = YOLO(path_yolo)
        # Monkey patch method
        model.model._predict_once = MethodType(_predict_once, model.model)

        # Load the FPN output layers
        #model.model.yaml # Find the FPN output layers
        _ = model(ASSETS / "bus.jpg", save=False, embed=[15, 18, 21, 22])
        
        return model

    def get_object(self, img_path):
        # Load image
        img = cv2.imread(img_path)

        # Preprocess and run inference
        prepped = self.model.predictor.preprocess([img])
        result = self.model.predictor.inference(prepped)

        # Apply non-max suppression
        output, idxs = non_max_suppression(result[-1][0], in_place=False)

        # Extract object features
        obj_feats = get_object_features(result[:3], idxs[0].tolist())
        output[0][:, :4] = scale_boxes(prepped.shape[2:], output[0][:, :4], img.shape)  # Convert to x1, y1, x2, y2, conf, class_id format

        # Compile results
        result = Results(img, path="", names=self.model.predictor.model.names, boxes=output[0])
        result.feats = obj_feats

        return result
