import torch
from torchvision import models
from problem_enum import Problem

class AGIModel:
    def __init__(self, selected_problems):
        self.problems = selected_problems
        self.object_detection_model = None
        self.image_classification_model = None
        self.initialize_models()
    
    def initialize_models(self):
        if Problem.OBJECT_DETECTION.value in self.problems:
            self.object_detection_model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
        if Problem.IMAGE_CLASSIFICATION.value in self.problems:
            self.image_classification_model = models.mobilenet_v3_small()