""" animal_classification.py """

# Import the Necessary Libraries
import os
import sys
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from PIL import Image
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer,
)

# Add the modules directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.preparedata import prepare_data
from modules.animalclasses import animal_classes


# Define data path
DATA_PATH = "./animaldata/images"

# Initialize empty lists for image paths and labels
labels = []
image_paths = []

# Prepare Data
# This is done in the preparedata module

# Prepare data for each animal class
for index, animal in enumerate(animal_classes):
    animal_image_paths, animal_labels = prepare_data(
        os.path.join(DATA_PATH, animal), index
    )
    if animal_image_paths is not None and animal_labels is not None:
        image_paths.extend(animal_image_paths)
        labels.extend(animal_labels)

# Convert labels to PyTorch tensors
labels = torch.tensor(labels).long()

# Split data into training, validation, and testing sets
NUM_CLASSES = len(animal_classes)
x_temp, x_test, y_temp, y_test = train_test_split(
    image_paths, labels, test_size=0.3, stratify=labels
)
x_train, x_val, y_train, y_val = train_test_split(
    x_temp, y_temp, test_size=0.5, stratify=y_temp
)


# Define dataset class
class AnimalDataset(Dataset):
    """Custom Dataset class for loading animal images and labels."""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert("RGB")
        img = img.resize((224, 224))  # Resize images to 224x224
        if self.transform:
            img = self.transform(img)
        return {
            "pixel_values": img,  # Key expected by the Trainer
            "label": label,  # Key for the label
        }


# Define transformations
train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Create datasets
train_dataset = AnimalDataset(x_train, y_train, transform=train_transform)
val_dataset = AnimalDataset(x_val, y_val, transform=val_test_transform)
test_dataset = AnimalDataset(x_test, y_test, transform=val_test_transform)

# Create data loaders with multiple workers for faster data loading
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

# Load pre-trained ViT model and modify it
MODEL_NAME = "google/vit-base-patch16-224"
feature_extractor = ViTImageProcessor.from_pretrained(MODEL_NAME)
model = ViTForImageClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_CLASSES, ignore_mismatched_sizes=True
)

# Set the id2label in the model's config
model.config.id2label = dict(enumerate(animal_classes))


# Function to compute accuracy
def compute_metrics(p):
    """Compute accuracy for model evaluation."""
    logits, labels = p
    pred = np.argmax(logits, axis=1)
    accuracy = (pred == labels).mean()
    return {"accuracy": accuracy}


# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=20,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate the model on test data
test_results = trainer.evaluate(eval_dataset=test_dataset)
print(f"Test Results: {test_results}")

# Save the final model
model.save_pretrained("./final_model")
feature_extractor.save_pretrained("./final_model")
