# src/part1DeepLearning/evaluation.py
# --------------------------------------------------
# model.py is responsible for defining the neural network architecture for vehicle detection.
# It is also responsible for creating a model and return a trained model.
# --------------------------------------------------
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# ==================================================
# VehicleDetector Class
# ------------------------------------------------
# __init__: Initializes the class with the number of classes for detection.
# get_model: Loads a pretrained Faster R-CNN model, modifies the head to match the number of classes, and returns the model.
# if __name__ == "__main__": This block allows the script to be run directly, creating an instance of VehicleDetector, getting the model, and printing it.
# ==================================================
class VehicleDetector:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def get_model(self):
        # Load pretrained model
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights="DEFAULT"
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features,
            self.num_classes
        )
        return model


if __name__ == "__main__":
    num_classes = 2  
    detector = VehicleDetector(num_classes)
    model = detector.get_model()

    print(model)
