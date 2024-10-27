#!/usr/bin/env python
# Class to load cat breed dataset with images, annotations and xml metadata.

import os, torch
import xml.etree.ElementTree as ET
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from collections import Counter

class CatDataset(Dataset):
    def __init__(self, root_dir, annotations_dir, transform=None, debug=False):
        self.root_dir = root_dir
        self.transform = transform
        self.annotations_dir = annotations_dir
        self.image_paths, self.labels = self.load_image_paths_and_labels()

        # Convert all labels to strings to avoid sorting errors
        self.labels = [str(label) for label in self.labels]
        self.classes = sorted(list(set(self.labels)) + ["Unknown"])  # Ensure "Unknown" is included in classes

        # Check number of occurrences of each label.
        if debug:
            label_counts = Counter(self.labels)
            print(f"Class Distribution: {label_counts}")

    def load_image_paths_and_labels(self):
        image_paths = []
        labels = []

        # Traverse images directory and extract breed names from filenames
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_name = os.path.splitext(file)[0]
                    file_path = os.path.join(root, file)
                    image_paths.append(file_path)

                    # Extract breed name based on capitalized parts of filename
                    breed_name = self.extract_breed_name(img_name)
                    labels.append(breed_name)

        return image_paths, labels

    def extract_breed_name(self, img_name):
        # Split the filename and extract parts that start with a capital letter
        parts = img_name.split('_')
        breed_name_parts = [part for part in parts if part and part[0].isupper()]

        # Join parts to form the breed name
        breed_name = "_".join(breed_name_parts)

        # Special case: set to "Unknown" if breed_name is "IMG" or empty
        if breed_name == "IMG" or not breed_name:
            return "Unknown"

        return breed_name

    def parse_xml(self, xml_path, width, height):
        # Parse XML file for bounding box if available
        boxes = []
        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for obj in root.findall("object"):
                bbox = obj.find("bndbox")
                xmin = int(bbox.find("xmin").text)
                ymin = int(bbox.find("ymin").text)
                xmax = int(bbox.find("xmax").text)
                ymax = int(bbox.find("ymax").text)
                boxes.append([xmin, ymin, xmax, ymax])
        else:
            boxes.append([0, 0, width, height])
        return boxes

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        class_name = self.labels[idx]
        label = self.classes.index(class_name)  # Convert class name to class index

        # Print for verification
        #print(f"Image: {img_path}, Label: {class_name}, Label Index: {label}")

        # Load image
        image = Image.open(img_path).convert("RGB")
        width, height = image.size  # Get image dimensions for placeholder box

        # XML annotation path
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        xml_path = os.path.join(self.annotations_dir, img_name + '.xml')

        # Parse bounding boxes if XML exists
        #print(f"XML file found for {xml_path}")
        boxes = self.parse_xml(xml_path, width, height)

        # Convert boxes to a fixed-size tensor for batching
        max_boxes = 5  # Set this based on expected max number of boxes per image
        padded_boxes = torch.zeros((max_boxes, 4))
        for i, box in enumerate(boxes[:max_boxes]):
            padded_boxes[i] = torch.tensor(box)


        if self.transform:
            image = self.transform(image)

        return image, label, padded_boxes

