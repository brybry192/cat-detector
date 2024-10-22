#!/usr/bin/env python

import os, sys, torch, torchvision 
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
# Load the pre-trained model (using a smaller model for performance)
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights


#class DetectionOnlyModel(torch.nn.Module):
#    def __init__(self, model):
#        super().__init__()
#        self.model = model
#    def forward(self, x):
#        _, detections = self.model(x)
#        return detections

# Use to see what models and weights are available.
def print_weights():
    import torchvision.models.detection as models
    print(dir(models))

# Run to check whether GPU or CPU is unsed for inference processing.
def check_gpu_support():
    if torch.cuda.is_available():
        print(f"CUDA is available. GPU device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Using CPU.")

# Other object detection models tried:
#     - fasterrcnn_mobilenet_v3_large_fpn     (Faster R-CNN with MobileNet backbone)
#     - fasterrcnn_mobilenet_v3_large_320_fpn (Lower resolution varient)
def load_model(model_path='models/ssdlite320_mobilenet_v3_large_dicts.pth'):
    # Define model architecture
    model = ssdlite320_mobilenet_v3_large(weights=None, num_classes=91, weights_backbone=None)

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, weights_only=True)
        model.load_state_dict(state_dict)
    else:
        # Load the pre-trained model
        model = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
        # Quantize from int32 to int8 to improve performance (reduces detection accuracy)
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        # export jit script for use with python or c++ or go
        script_model = torch.jit.script(model)
        script_model.save("models/ssdlite320_mobilenet_v3_large_dicts.pt")
        # Save the model for future use
        torch.save(model.state_dict(), model_path)
        print(f"Model dicts downloaded and saved locally to {model_path}")
    model.eval()
    return model


def process_image(image_path):

    # Load the image
    image = Image.open(image_path).convert('RGB')

    # Resize image proportionally
    # Resize the image to improve detection performance.
    max_size = (600, 600)
    image.thumbnail(max_size, Image.LANCZOS)

    #detection_only_model = DetectionOnlyModel(model)
    # Convert image to tensor
    image_tensor = F.to_tensor(image).to(device)
    #script_model = torch.jit.trace(detection_only_model, [image_tensor])
    #script_model.save("models/ssdlite320_mobilenet_v3_large_trace.pt")
    #image_tensor = image_tensor.unsqueeze(0)

    # Perform detection
    with torch.no_grad():
        predictions = model([image_tensor])


    # Move predictions to CPU if necessary
    predictions = [{k: v.to('cpu') for k, v in t.items()} for t in predictions]
    
    # Extract labels, boxes, and scores
    labels = predictions[0]['labels']
    boxes = predictions[0]['boxes']
    scores = predictions[0]['scores']
    
    # Identify cat predictions/detections (class ID 17 for cats in COCO dataset)
    cat_indices = (labels == 17) & (scores > 0.5)  # Adjust confidence threshold as needed
    
    # Extract cat boxes
    cat_boxes = boxes[cat_indices]
    
    # Draw bounding boxes on the image
    draw = ImageDraw.Draw(image)
    for box in cat_boxes:
        draw.rectangle(box.tolist(), outline='red', width=3)
    
    # Show or save the image with detections
    image.show()
    #image.save('cat_detection_output.jpg')

if __name__ == "__main__":

    #check_gpu_support()
    #os.environ['TORCH_HOME'] = '/Users/bryant/.cache/torch/hub/checkpoints'
    
    # Load the model once
    model = load_model()

    # Optionally move processing to GPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Process multiple images in a loop
    image_paths = sys.argv[1:]
    for image_path in image_paths:
        process_image(image_path)

