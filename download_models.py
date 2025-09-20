import torch
from ultralytics import YOLO
import os

# Folder to save models
os.makedirs("models", exist_ok=True)

# --------------------------
# DETR Models
# --------------------------
print("Starting DETR downloads...")

print("Downloading DETR ResNet-50...")
detr50 = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
torch.save(detr50.state_dict(), "models/detr50.pth")
print("DETR50 saved to models/detr50.pth ✅")

print("Downloading DETR ResNet-101...")
detr101 = torch.hub.load('facebookresearch/detr', 'detr_resnet101', pretrained=True)
torch.save(detr101.state_dict(), "models/detr101.pth")
print("DETR101 saved to models/detr101.pth ✅")

# --------------------------
# YOLOv8 Models
# --------------------------
print("\nStarting YOLOv8 downloads...")

yolo_models = {
    "YOLOv8x": "yolov8x.pt",
    "YOLOv8l": "yolov8l.pt"
}

for name, filename in yolo_models.items():
    print(f"Downloading {name}...")
    model = YOLO(filename)  # Ultralytics downloads if not present
    model.save(f"models/{filename}")
    print(f"{name} saved to models/{filename} ✅")

print("\nAll models downloaded and saved in ./models folder 🎉")
