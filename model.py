import torch
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms
from problem_enum import ProblemType
from constants import COCO_INSTANCE_CATEGORY_NAMES

class AGIModel:
    def __init__(self, detected_problems):
        self.problems = detected_problems
        self.object_detection_model = None
        self.image_classification_model = None
        self.initialize_models()
    
    def initialize_models(self):
        if ProblemType.OBJECT_DETECTION.value in self.problems:
            self.object_detection_model = ObjectDetectionModel()
        if ProblemType.IMAGE_CLASSIFICATION.value in self.problems:
            self.image_classification_model = ImageClassificationModel()
    
    def get_results(self, input_path):
        results_dct = {ProblemType.OBJECT_DETECTION.value: None,
                       ProblemType.IMAGE_CLASSIFICATION.value: None}
        if self.object_detection_model is not None:
            obj_detection_pred = self.object_detection_model.get_inference(input_path)
            results_dct[ProblemType.OBJECT_DETECTION.value] = obj_detection_pred
        if self.image_classification_model is not None:
            img_classification_pred = self.image_classification_model.get_inference(input_path)
            results_dct[ProblemType.IMAGE_CLASSIFICATION.value] = img_classification_pred
        return results_dct
            
        

class ObjectDetectionModel:
    def __init__(self):
        self.model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT")
    
    def get_inference(self, input_path):
        image = Image.open(input_path).convert("RGB")
        img_tensor = self.transform(image)
        self.model.eval()
        pred = self.model([img_tensor])
        scores = pred[0]["scores"].cpu().detach().numpy()
        boxes = pred[0]["boxes"].cpu().detach().numpy()[scores > 0.5]
        labels = pred[0]["labels"].cpu().detach().numpy()[scores > 0.5]
        labels = [COCO_INSTANCE_CATEGORY_NAMES[l] for l in labels]
        scores = scores[scores > 0.5]
        result = {"boxes": boxes , "labels": labels, "scores": scores}
        return result 
    
    def transform(self, image):
        transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float32)
        ])
        return transform(image)
        
class ImageClassificationModel:
    def __init__(self):
        self.model = models.mobilenet_v3_large(weights="MobileNet_V3_Large_Weights.DEFAULT")
        
    def get_inference(self, input_path):
        image = Image.open(input_path).convert("RGB")
        img_tensor = self.transform(image)
        self.model.eval()
        pred = self.model(torch.unsqueeze(img_tensor, 0))
        percentage = torch.nn.functional.softmax(pred, dim=1)[0] * 100
        pred = torch.argmax(percentage).cpu().detach().numpy()
        confidence = percentage[pred].detach().numpy()
        with open("sample_data/imagenet_classes.txt", "r") as fp:
            classes = fp.readlines()
        print(classes)
        pred = classes[pred].replace("\n","")
        return pred, confidence
    
    def transform(self, image):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        return transform(image)