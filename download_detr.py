import torch
import torchvision

# DETR with ResNet-50 backbone
detr50 = torchvision.models.detection.detr_resnet50(pretrained=True)
detr50.eval()  # Set to evaluation mode

# DETR with ResNet-101 backbone
detr101 = torchvision.models.detection.detr_resnet101(pretrained=True)
detr101.eval()
