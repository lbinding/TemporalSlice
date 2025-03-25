#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 01 09:33:30 2025

@author: lawrencebinding
"""


#%% Libs 
import torch
import numpy as np
import torchio as tio
from torch.utils.data import DataLoader
from temporalSlice.model import temporalSliceNet
from temporalSlice.utils import get_device
import nibabel as nib
import argparse
import os 
import warnings
import scipy.ndimage as ndi
warnings.simplefilter("ignore", RuntimeWarning)

#%% Crop / Resizing initialisation
rescale = tio.RescaleIntensity((-1, 1))
target_shape = 192, 192, 192
crop_pad = tio.CropOrPad(target_shape)
transform = tio.Compose([crop_pad, rescale])
device = get_device()

#%% Function 
def inference(T1_file, output_file):
    #Load data in 
    T1_nii = nib.load(T1_file)
    T1_data = T1_nii.get_fdata()
    
    #Create a subject (needed for reverse resizing/cropping)
    subject = tio.Subject(t1=tio.ScalarImage(tensor=torch.from_numpy(T1_data[None,:,:,:]).to(dtype=torch.double)),
                          mask=tio.ScalarImage(tensor=torch.from_numpy(T1_data[None,:,:,:]).to(dtype=torch.double)))
    
    #Torchio requirements 
    subjects_dataset = tio.SubjectsDataset([subject], transform=transform)
    training_loader = DataLoader(
        subjects_dataset,
        collate_fn=tio.utils.history_collate,
    )

    #Load out model
    model, _, _ = temporalSliceNet()

    for subjects_batch in training_loader:
        #Get subjects from batch 
        tio_subj = tio.utils.get_subjects_from_batch(subjects_batch)[0]
        #Convert to double / unsqueeze 
        T1_tensor = tio_subj['t1'].data.clone().detach().to(dtype=torch.double).unsqueeze(0).to(device)

        # Set model to evaluation mode
        model.eval()
        
        # Disable gradients for inference
        with torch.no_grad():
            logits = model(T1_tensor)  # Forward pass
            probs = torch.sigmoid(logits)  # Convert logits to probabilities
        
        #Threshold
        binary_mask = (probs > 0.5).float()  # Convert to binary (0 or 1)
        
        #Get largest connected component
        binary_mask = get_largest_component(binary_mask)

        #Inject back into the subject 
        tio_subj.mask.set_data(binary_mask.squeeze(0))
        
        #Inverse the transformations 
        tio_subj = tio_subj.apply_inverse_transform()

        # Save prediction as a NIfTI file
        pred_nii = nib.Nifti1Image(tio_subj.mask.data.squeeze().numpy().astype(np.uint8), affine=T1_nii.affine)
        nib.save(pred_nii, output_file)

        #
        print("Successful segmentation")

#%% Helper function
def get_largest_component(binary_mask):
    labeled_array, num_features = ndi.label(binary_mask.cpu().numpy())  # Label connected components
    if num_features == 0:
        return binary_mask  # No components found, return original
    
    sizes = ndi.sum(binary_mask.cpu().numpy(), labeled_array, range(1, num_features + 1))
    largest_component = (labeled_array == (sizes.argmax() + 1))  # Keep only the largest component
    
    return torch.tensor(largest_component, dtype=binary_mask.dtype, device=binary_mask.device)


#%% Parse information 
parser = argparse.ArgumentParser()
parser.add_argument('--in', dest='T1_file', help='Input T1 image', required=True)
parser.add_argument('--out', dest='out_file', help='Output resection mask', required=True)
args = parser.parse_args()

# call function to do correction
inference(os.path.abspath(args.T1_file), os.path.abspath(args.out_file))