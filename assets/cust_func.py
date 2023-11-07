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
import torch.nn as nn
from torchvision import models
import argparse

def read_smi_file(file_path):
    try:
        with open(file_path, 'r') as smi_file:
            lines = smi_file.readlines()
        return [line.strip() for line in lines]
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []

def smi_to_dataframe(file_path):
    smi_data = read_smi_file(file_path)
    if not smi_data:
        return None
    
    data = {'SMILES': smi_data}
    df = pd.DataFrame(data)
    return df
# Define a function to preprocess SMILES strings and generate images
def smi_to_image(smiles, size=(224, 224)):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        img = Draw.MolToImage(mol, size=size)
        img = img.convert('RGB')
        return img
    return None
    
# Create a custom dataset class for loading SMILES and converting them to images
class SMILESDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        smiles = self.df.iloc[idx]['SMILES']
        image = smi_to_image(smiles)
        if self.transform:
            image = self.transform(image)
        return image
        
# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images
    transforms.ToTensor(),  # Convert to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

