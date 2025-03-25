# TemporalSlice – Temporal lobe resection segmentation

Developed by: [lawrence.binding.19@ucl.ac.uk](mailto:lawrence.binding.19@ucl.ac.uk)

## Introduction

TemporalSlice is a model to segment the resection cavity of postoperative T1 images following a temporal lobe resection. It is trained on completely synthetic data with good generalisability (paper in progress...). Feel free to try and test below.  

## Requirements

- **Operating System**: Mac or Linux
- **Freesurfer**: Version 7.4 or higher
- **Python**: Installed

## Installation

1. **Download the repository** and place it in a desired directory. For example, you can place it at `/Users/lawrence/` (this path will be referred to as `<path>`).

2. **Set the script path**: 
   After downloading the repository, you need to add the path to the `scripts` directory in your `bashrc` (LINUX). If you're on MacOS you'll need to either update your `bash_profile` or `zshrc` depending on if you're using bash or zsh in the terminl window. By default, mac uses `zshrc`. Replace `bashrc` in the following code with the desired shell target. Replace `<path>` with the actual path where the software is installed:

```bash
echo 'export PATH="<path>/TemporalSlice/:$PATH"' >> ~/.bashrc
```
For example, if you installed the software in /Users/lawrence/:
```bash
echo 'export PATH="/Users/lawrence/TemporalSlice:$PATH"' >> ~/.bashrc
```

Source the updated bashrc:
To apply the changes, either close and reopen your terminal or run:

```bash
source ~/.bashrc
```

Install the required Python packages:
Navigate to the repository directory and install the required Python dependencies via pip:

```bash
pip install -r requirements.txt
```

This will install all necessary packages for running the scripts.

## Usage
Once everything is set up, you can call the scripts directly from your terminal. Here’s an example:

```bash
TemporalSlice.sh -in T1_postop.nii.gz -out T1_postop_seg.nii.gz
```

Notes:
In the current model the T1 needs to be registered to MNI152_T1_1mm.nii.gz, this can be done using mri_easyreg, this is subject to change. Please reach out if this is a requirement to get the latest, aquisition agnostic models. 

