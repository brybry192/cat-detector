#!/usr/bin/env python
# Perform custom training on top of an existing model, adding ability to detect the cat breed.
# The cats and annotations downloaded from https://www.robots.ox.ac.uk/~vgg/data/pets/ plus
# various pictures of our tabby cat were used to train and refine the model.
# Following the format described in lists.txt, all tabby pictures were prefixed with Tabby_.
# Save custom weights after training completes for use with cat breed detection.

import argparse, os, time, torch
from cat_breed_dataset import CatDataset  # CatDataset is defined in cat_breed_dataset.py
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision.transforms import v2


# Set up argument parser that can be used to define paths used for training, validation and annotations.
parser = argparse.ArgumentParser(description="Detect the cat breed(s) in images.")
parser.add_argument("--train_dir", default="data/train", type=str, help="Directory path with jpg images used for training.")
parser.add_argument("--val_dir", default="data/val", type=str, help="Directory path with jpg images used for validation.")
parser.add_argument("--annotations_dir", default="data/annotations/xmls", type=str, help="Directory path with the matching xml annotations used for training.")
parser.add_argument("--save_model_weights", default="models/cat_breed_resnet50.pth", type=str, help="After training save the finely tuned model weights for cat breed detection to this path.")
args = parser.parse_args()


# Define transformations for the training and validation datasets.
data_transforms = {
    'train': v2.Compose([
        v2.Resize((224, 224)),                                         # Ensure final size consistency
        v2.RandomResizedCrop(224, scale=(0.8, 1.0)),                   # Random crop with consistent output size
        v2.RandomHorizontalFlip(),                                     # Random horizontal flip
        v2.RandomAutocontrast(0.1),
        #v2.ElasticTransform(alpha=50.0), # Random transforms the morphology produces a see-through-water-like effect.
        v2.RandomRotation(degrees=(0, 60)),                            # Random rotation between 0-60 degress
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Strong color jitter
        #v2.RandomAffine(degrees=15, translate=(0.1, 0.1), shear=10),  # Random affine transform
        #v2.RandomPerspective(distortion_scale=0.5, p=0.5),            # Perspective distortion
        #v2.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)), # Random erasing
        v2.ToTensor(),
        #v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Normalize as per ImageNet
    ]),
    'val': v2.Compose([
        v2.Resize((224, 224)),  # Resize to a consistent size for validation
        v2.ToTensor(),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}


# Load datasets with CatDataset with transforms from cat_dataset.py.
image_datasets = {
    'train': CatDataset(root_dir=args.train_dir, annotations_dir=args.annotations_dir, transform=data_transforms['train'], debug=True),
    'val': CatDataset(root_dir=args.val_dir, annotations_dir=args.annotations_dir, transform=data_transforms['val'], debug=True)
}
dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
        'val': DataLoader(image_datasets['val'], batch_size=32, shuffle=False)
}


# Support GPU cuda use if available on system.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Load a pre-trained model and configure with the number of features and class names.
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
num_features = model.fc.in_features
class_names = image_datasets['train'].classes
#model.fc = nn.Linear(num_features, len(class_names))  # Adjust the final layer to match the number of classes.

# Modify the fully connected layer with dropout
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),                  # Dropout layer with 50% dropout rate
    nn.Linear(model.fc.in_features, len(class_names))  # Fully connected layer
)

# Move to GPU if available.
model = model.to(device)
# Set up the loss function and optimizer.
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.8)  # Initial learning rate
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
num_epochs = 10

training_start = time.time()
print(f"Starting training on {device}\n")


# Define headers for table view of progress and format.
headers = ["Epoch", "Duration (s)", "Phase", "Loss", "Accuracy"]
header_format = "{:<8} {:<12}\t{:<8} {:<8} {:<8}"
row_format = "{:<8} {:<12.2f}\t{:<8} {:<8.4f} {:<8.4f}"

# Print headers
print(header_format.format(*headers))
print("-" * 60)

for epoch in range(num_epochs):
    for phase in ['train', 'val']:
        start_time = time.time()  # Start timer for epoch
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels  in dataloaders[phase]: #, boxes in dataloaders[phase]:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Use bounding boxes in the training process
            with torch.set_grad_enabled(phase == 'train'):
                # Forward pass: compute model outputs
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(image_datasets[phase])
        epoch_acc = running_corrects.double() / len(image_datasets[phase])

        phase_duration = time.time() - start_time  # Calculate epoch duration
        row = [f"{epoch+1}/{num_epochs}", phase_duration, phase, epoch_loss, epoch_acc]
        print(row_format.format(*row))

        # Update the scheduler based on validation loss
        scheduler.step(epoch_loss)

training_duration = time.time() - training_start  # Calculate training duration
print(f"\nTraining complete in {training_duration:.2f} seconds")

# Save the finely tuned model weights. 
torch.save(model.state_dict(), args.save_model_weights)
print(f"Model saved as {args.save_model_weights}")

