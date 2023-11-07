import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Draw
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
from io import BytesIO
from torchvision.models import mobilenet_v2
import torch.nn as nn
from torchvision import models
from assets.cust_func import read_smi_file, smi_to_dataframe, smi_to_image, SMILESDataset, transform
import argparse         
pretrained_model = mobilenet_v2(pretrained=True)

# Modify the classifier (fully connected) layers for your specific task
num_classes = 2  # Change this to the number of classes in your binary classification task

# Access the classifier part of the model (different from 'fc' attribute in other models)
classifier = nn.Sequential(
    nn.Dropout(0.2),  # Optional dropout layer
    nn.Linear(pretrained_model.last_channel, num_classes)
)

# Set the classifier as the final layers of the model
pretrained_model.classifier = classifier
# Load the state dictionary
checkpoint = torch.load('assets/pytorchModelCNN.pth')

# Load the state dictionary into your modified model
pretrained_model.load_state_dict(checkpoint)
# Set the model to evaluation mode
pretrained_model.eval()
# Make predictions
predictions = []

# Create an argument parser
parser = argparse.ArgumentParser(description='Converting a ".smi" file to a pandas DataFrame.')

# Add an argument for the path to the test.smi file
parser.add_argument('input_file', type=str, help='Path to the .smi file')

# Parse the command-line arguments
args = parser.parse_args()
input_file = args.input_file

df = smi_to_dataframe(input_file)

# Create a DataLoader for the test SMILES
#test_df = pd.read_csv('test.smi', names=['SMILES'], delimiter='\t')
test_dataset = SMILESDataset(df, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=8)

with torch.no_grad():
    for images in tqdm(test_loader, desc='Predicting'):
        outputs = pretrained_model(images)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.cpu().numpy())
        
# Interpret the model's predictions
for i, prediction in enumerate(predictions):
    if prediction == 0:
        print(f'Molecule {i + 1} is predicted to be Inactive')
    else:
        print(f'Molecule {i + 1} is predicted to be Active')
        
        

