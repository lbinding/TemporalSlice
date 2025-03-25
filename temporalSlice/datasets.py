import torch
from torch.utils.data import Dataset
import nibabel as nib
import torchio as tio
from pathlib import Path
from temporalSlice.syntheticResection import SyntheticResection  # Assuming the class exists
from temporalSlice.augmentor import Augmentor  # Assuming the class exists
import random

class BrainDataset(Dataset):
    def __init__(self, data_dir, augment):
        self.T1_dir = Path(data_dir, 'T1scans')
        self.GIF_dir = Path(data_dir, 'GIFsegmentation')
        self.T1_paths = sorted(self.T1_dir.glob('*.nii.gz'))
        self.GIF_paths = sorted(self.GIF_dir.glob('*.nii.gz'))
        self.augment = augment

        assert len(self.T1_paths) == len(self.GIF_paths), "Mismatch in number of T1 and GIF scans"

    def __len__(self):
        return len(self.T1_paths)

    def __getitem__(self, idx):
        T1_data = nib.load(self.T1_paths[idx]).get_fdata()
        GIF_data = nib.load(self.GIF_paths[idx]).get_fdata()

        T1_data_rm, resection_mask = SyntheticResection(T1_data, GIF_data).create_resection()

        subject = tio.Subject(
            mask=tio.LabelMap(tensor=torch.from_numpy(resection_mask[None,:,:,:])),
            t1=tio.ScalarImage(tensor=torch.from_numpy(T1_data_rm[None,:,:,:]))
        )

        #Perform augmentations 
        if self.augment: 
            aug_level = random.randint(1, 3)
            subject_aug = Augmentor().apply_augmentations(subject, aug_level)
        else:
            subject_aug = Augmentor().apply_augmentations(subject, 0)

        return subject_aug

def get_datasets(train_dir, val_dir):
    train_dataset = BrainDataset(train_dir, augment=True)
    val_dataset = BrainDataset(val_dir, augment=False)
    return train_dataset, val_dataset
