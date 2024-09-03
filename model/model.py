"""
model.py

This module defines the model and associated functions for animal image classification
using a pre-trained HuggingFace Transformers model. It includes preprocessing of images
and running predictions to classify the animal present in the image.
"""

import os
import io
import logging
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch
from modules.animalnames import animal_names

# Initialize the model and tokenizer
MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "final_model")
)

feature_extractor = ViTImageProcessor.from_pretrained(MODEL_PATH)
model = ViTForImageClassification.from_pretrained(MODEL_PATH)

try:
    feature_extractor = ViTImageProcessor.from_pretrained(MODEL_PATH)
    model = ViTForImageClassification.from_pretrained(MODEL_PATH)
    print("Model loaded successfully!")
except ImportError as e:
    print(f"Error loading model: {e}")


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """
    Preprocesses the input image by converting it to RGB and tokenizing it for the model.

    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    return inputs["pixel_values"]


def predict(image_bytes: bytes) -> str:
    """Makes a prediction on the input image and returns the corresponding animal class."""
    try:
        # Preprocess image
        inputs = preprocess_image(image_bytes)

        # Model inference
        with torch.no_grad():  # Ensure no gradients are computed
            outputs = model(inputs)  # Pass preprocessed image tensor directly

        # Extract prediction
        logits = outputs.logits
        predicted_class_id = logits.argmax(-1).item()
        return animal_names[predicted_class_id]
    except KeyError as e:
        logging.error("KeyError in accessing class index: %s", e)
        return "Error during Accessing class index"
    except IndexError as e:
        logging.error("IndexError in class prediction: %s", e)
        return "Error during Class Prediction"
    except RuntimeError as e:
        logging.error("RuntimeError during model prediction: %s", e)
        return "Error during model prediction"
    except ImportError:
        logging.error("Unexpected error during prediction: %s", e)
        return "Unexpected error during prediction"
