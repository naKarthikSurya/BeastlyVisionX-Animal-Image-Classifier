# BeastlyVisionX: Animal Image Classifier

## Overview
BeastlyVisionX is a sophisticated AI-powered animal image classification system that accurately identifies and classifies over 279 different animal species from images. The system  leverages Transfer Learning by fine-tuning a pre-trained Vision Transformer (ViT) model, providing a robust and efficient solution for applications in wildlife monitoring, educational tools, and conservation efforts.

## Project Structure
- `Animal-Image-Classification/`
  - `animalclassification/`
    - `animal_classification.py`
  - `animaldata/`
    - `Images/`
      - `(Dataset of Various Animal Classes)`
  - `main/`
    - `__init__.py`
    - `main.py`
    - `templates/`
      - `home.html`
      - `model.html`
      - `features.html`
      - `about.html`
    - `static/`
      - `styles.css`
  - `model/`
    - `__init__.py`
    - `model.py`
  - `modules/`
    - `__init__.py`
    - `animalclasses.py`
    - `preparedata.py`
  - `final_model/`
    - `model.safetensors`


## Features
- **Animal Image Classification**: Classify images into various animal classes using pre-trained HuggingFace Transformers models.
- **Transfer Learning**: Utilizes a pre-trained Vision Transformer (ViT) model, significantly reducing the amount of training data and time required.
- **Custom Model Training**: Train a custom model on your dataset using HuggingFace Transformers.
- **High Accuracy**: Fine-tuned on a diverse dataset to ensure precise animal classification.
- **Web Application**: A user-friendly web interface built with FastAPI for uploading images and receiving predictions.

# Getting Started

### Prerequisites

- Python 3.8 or higher
- Required Python packages (see `requirements.txt`)

### Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/naKarthikSurya/Animal-Image-Classification.git
    cd Animal-Image-Classification
    ```

2. **Create a virtual environment and activate it:**

    ```bash
    python -m venv venv
    venv/bin/activate 
    ```

3. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

### Dataset Preparation

1. **Organize your dataset:**

    Place your dataset in the `animaldata/Images/` directory, with subfolders for each animal class. Each subfolder should contain images of the corresponding animal class.

2. **Preprocess the dataset**:

    the `modules/preparedata.py` script to preprocess the images which is used in the training script 

### Model Training

1. **Set up the training script:**

    Prepare and configure the `animalclassification/animal_classification.py` script to fine-tune a pre-trained HuggingFace Transformers model on your dataset.

2. **Run the training script:**

    ```bash
    python animalclassification/animal_classification.py
    ```

    Adjust the batch size, learning rate, and other hyperparameters according to your system's capabilities.

3. **Save the trained model:**

    After training, save the model in the `final_model/` directory:

    ```bash
    # Save the final model
    model.save_pretrained("./final_model")
    feature_extractor.save_pretrained("./final_model")
    ```

### Model Inference

1. **Run the FastAPI application:**

    ```bash
    uvicorn main.main:app --reload
    ```

2. **Access the web interface:**

    Open your web browser and navigate to `http://127.0.0.1:8000`. You can upload images and view classification results.

### Usage

1. **Upload an image:**

    Use the web interface to upload an image for classification.

2. **View the classification:**

    The system will classify the image into one of the animal classes and provide a detailed description.

