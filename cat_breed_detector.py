#!/usr/bin/env python
# Use ResNet50 model with custom weights for detecting the breed of cat in images.

import argparse, glob, os, torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify

# Initialize Flask app for http server.
app = Flask(__name__)

# Initialize weights for new layers only
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# Function to predict if a tabby cat is in the image.
def detect_cat_breed(image_path, threshold):

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
    msg = f"{predicted_class_name} detected in {image_path} with probability {confidence:.2f}"

    return msg, predicted_class_name, confidence

def detect_tabby_cat(image_path, threshold):
    """Function to detect if a tabby cat is in a photo"""
    # Load the image and apply transformations
    image = Image.open(image_path).convert('RGB')
    image = data_transforms(image).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        output = tabby_model(image)
        probabilities = F.softmax(output, dim=1)
        _, predicted_class = torch.max(output, 1)

    # Get class name using predicted index
    predicted_class_name = tabby_class_names[predicted_class.item()]
    confidence = probabilities[0][predicted_class.item()].item()

    # Report cat breed when confidence in detection is high enough.
    msg = f"{predicted_class_name} detected in {image_path} with probability {confidence:.2f}"

    return msg, predicted_class_name, confidence

@app.route('/detect_cat', methods=['POST'])
def detect_cat():
    if 'file' in request.files:
        # Handle file upload
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        image = Image.open(file.stream)
        image_path = "/tmp/uploaded_image.jpg"
        image.save(image_path)
    elif 'image_path' in request.json:
        # Handle image path in JSON
        image_path = request.json['image_path']
        if not os.path.exists(image_path):
            return jsonify({'error': 'Image path does not exist'}), 400
    else:
        return jsonify({'error': 'No file or image path provided'}), 400

    # Run detection
    is_tabby = False
    msg, resp, tabby_confidence = detect_tabby_cat(image_path, args.confidence)
    if resp == "tabby_cat":
        is_tabby = True
    msg, breed, breed_confidence = detect_cat_breed(image_path, args.confidence)
 
    return jsonify({'message': msg, 'breed': breed, 'breed_confidence': breed_confidence, 'is_tabby': is_tabby, 'is_tabby_confidence': tabby_confidence})

#
# Set up argument parser that can be used to define paths used for directories.
parser = argparse.ArgumentParser(description="Detect the cat breed(s) in images.")
parser.add_argument("--confidence", default=0.50, type=float, help="The confidence threshold for detecting cat breed or not cat in the image.")
parser.add_argument("--http_port", default=8081, type=int, help="The HTTP Port to run the web server on.")
parser.add_argument("--breed_model", default="models/cat_breed_classification_resnet50.pth", type=str, help="Path to the model file with custom weights for breed detection.")
parser.add_argument("--tabby_model", default="models/tabby_cat_resnet50.pth", type=str, help="Path to the model file with custom weights for tabby pattern detection.")
parser.add_argument("input_path", nargs="?", type=str, help="Path to jpg image or a directory of images.")
args = parser.parse_args()



class_names = [
        'Abyssinian', 'American_Shorthair', 'Bengal', 'Bombay', 'British_Shorthair',
        'Exotic_Shorthair', 'Maine_Coon', 'No_Cat', 'Persian', 'Ragdoll', 'Russian_Blue',
        'Scottish_Fold', 'Siamese', 'Sphynx'
]


# Define the base model architecture and structure.
model = models.resnet50(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(model.fc.in_features, len(class_names))
)
model.fc[1].apply(initialize_weights)

# tabby model
tabby_class_names = [ 'cat', 'no_cat', 'tabby_cat' ]
# Define the tabby base model architecture and structure.
tabby_model = models.resnet50(weights=None)
tabby_model.fc = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(tabby_model.fc.in_features, len(tabby_class_names))
)
tabby_model.fc[1].apply(initialize_weights)


# Support GPU cuda use if available on system.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the saved state_dict for custom model to detect tabby or not.
model.load_state_dict(torch.load(args.breed_model, map_location=torch.device(device), weights_only=True))
model.eval()

# Load and evalute tabby model.
tabby_model.load_state_dict(torch.load(args.tabby_model, map_location=torch.device(device), weights_only=True))
tabby_model.eval()


# Define the image transformation process to match what was used during training
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the input size expected by the model
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalization to ImageNet standards
])


if __name__ == '__main__':
    if args.input_path:
        images = glob.glob(args.input_path)
        if not args.input_path.endswith(('.jpg', '.jpeg', '.png')):
            images = glob.glob(f"{args.input_path}/*.jpg")
        # Loop through all images in the directory (you can adjust the pattern to match specific extensions)
        for image_path in images:
            # Run detection
            msg, tabby, tabby_confidence = detect_tabby_cat(image_path, args.confidence)
            msg, breed, breed_confidence = detect_cat_breed(image_path, args.confidence)
            print(f"{msg}")
    else:
        # Start HTTP server
        app.run(host='0.0.0.0', port=args.http_port)

