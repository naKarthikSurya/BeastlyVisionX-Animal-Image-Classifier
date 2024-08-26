"""
preparedata.py
This Module initialize the empty lists for image paths and labels Then Prepares the Data 
"""

import os


# Prepare Data
def prepare_data(path, class_index):
    """Prepare image paths and labels for each animal class."""
    image_paths = []
    labels = []
    animal_filenames = os.listdir(path)
    for filename in animal_filenames:
        img_path = os.path.join(path, filename)
        if os.path.isfile(img_path):
            image_paths.append(img_path)
            labels.append(class_index)
    return image_paths, labels
