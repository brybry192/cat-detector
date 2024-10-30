#!/usr/bin/env python
# Use ResNet50 model with custom weights for detecting the breed of cat in images.

import argparse, glob, os, torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from cat_breed_dataset import CatDataset

# Initialize weights for new layers only
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# Function to predict if a tabby cat is in the image.
def predict_image(image_path, threshold):

    # Load the image and apply transformations
    image = Image.open(image_path).convert('RGB')
    image = data_transforms(image).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        _, predicted_class = torch.max(output, 1)

    # Get class name using predicted index
    predicted_class_name = class_names[predicted_class.item()]
    confidence = probabilities[0][predicted_class.item()].item()

    # Report cat breed when confidence in detection is high enough.
    if confidence > threshold:
        print(f"{predicted_class_name} detected in {image_path} with probability {confidence:.2f}")

class_names = [ 'cat', 'no_cat', 'tabby_cat' ]

# Set up argument parser that can be used to define paths used for directories.
parser = argparse.ArgumentParser(description="Detect the cat breed(s) in images.")
parser.add_argument("--confidence", default=0.10, type=float, help="The confidence threshold for showing detected tabby, cat or not cat.")
parser.add_argument("--model_path", default="models/tabby_vs_non_tabby_resnet50.pth", type=str, help="Path to the modele file with custom weights for breed detection.")
parser.add_argument("input_path", default="images", type=str, help="Path to jpg image or a directory of images.")
args = parser.parse_args()

# Load the dataset to determine the number of classes
num_classes = len(class_names)  # Dynamically set number of classes based on dataset

# Define the base model architecture and structure.
model = models.resnet50(weights=None)
#model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.fc = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(model.fc.in_features, len(class_names))
)
model.fc[1].apply(initialize_weights)

# Support GPU cuda use if available on system.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the saved state_dict for custom model to detect tabby or not.
model.load_state_dict(torch.load(args.model_path, map_location=torch.device(device), weights_only=True))
model.eval()

# Define the image transformation process to match what was used during training
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the input size expected by the model
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalization to ImageNet standards
])

images = glob.glob(args.input_path)
if not args.input_path.endswith(('.jpg', '.jpeg', '.png')):
    images = glob.glob(f"{args.input_path}/*.jpg")


# Loop through all images in the directory (you can adjust the pattern to match specific extensions)
for image_path in images:
    predict_image(image_path, args.confidence)

