# src/part1DeepLearning/model.py
# --------------------------------------------------
# model.py is responsible for defining the neural network architecture for vehicle detection.
# It loads a pretrained faster R-CNN model & replace the classification head to adjust the no. of output classes.
# It returns a ready-to-train model that can be used in the training loop defined in train.py.
# --------------------------------------------------


# ==================================================
# Importing necessary libraries
# ------------------------------------------------
# Torch and Torchvision: For building and training the neural network model.
# ==================================================
import torchvision
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# ==================================================
# VehicleDetector Class
# ------------------------------------------------
# __init__: Initializes the class with the number of classes for detection.
# get_model: Loads a pretrained faster R-CNN model, replaces the classification head to match the number of classes, and returns the modified model.
# ==================================================
class VehicleDetector:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def get_model(self):

        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT

        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=weights
        )

        in_features = model.roi_heads.box_predictor.cls_score.in_features

        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features,
            self.num_classes
        )

        return model