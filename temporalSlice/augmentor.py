#%% Libs 
import random
import torchio as tio
import numpy as np

#%% Setup class 
class Augmentor:
    def __init__(self, voxel_size=1, target_shape=(192, 192, 192)):
        #Input voxel size and target shape 
        self.voxel_size = voxel_size
        self.target_shape = target_shape

        # Define augmentations
        self.random_anisotropy = tio.RandomAnisotropy()
        self.random_affine = tio.RandomAffine()
        self.add_motion = tio.RandomMotion(num_transforms=1, image_interpolation='nearest')
        self.rescale = tio.RescaleIntensity((-1, 1))
        self.crop_pad = tio.CropOrPad(self.target_shape, mask_name='mask')

    def blur(self, input_sub):
        """
        Applies a random Gaussian blur to simulate lower-resolution images.
        - Downsamples the image by a factor of 2 or 3 (randomly chosen).
        - Computes the standard deviation for the Gaussian blur based on the downsampling factor.
        """
        downsampling_factor = random.randint(2, 3)
        std = tio.Resample.get_sigma(downsampling_factor, self.voxel_size)
        antialiasing = tio.Blur(std)
        return antialiasing(input_sub)

    def anistropy(self, input_sub):
        """
        Introduces anisotropy (non-uniform resolution across different axes).
        - Simulates the effect of acquiring images with different voxel sizes.
        """
        return self.random_anisotropy(input_sub)

    def affine(self, input_sub):
        """
        Applies a random affine transformation to the image.
        - Includes random translation, rotation, scaling, and shearing.
        """
        return self.random_affine(input_sub)

    def elastix(self, input_sub):
        """
        Applies a random elastic deformation to simulate realistic anatomical variability.
        - Uses a grid of control points to warp the image non-linearly.
        - The displacement magnitude is randomly chosen in each direction (x, y, z).
        """
        max_displacement = random.randint(1, 5), random.randint(1, 5), random.randint(1, 5)
        random_elastic = tio.RandomElasticDeformation(
            max_displacement=max_displacement,
            num_control_points=random.randint(10, 20),
        )
        return random_elastic(input_sub)

    def noise(self, input_sub):
        """
        Adds random Gaussian noise to the image.
        - The standard deviation of the noise is randomly chosen.
        - Simulates scanner noise or acquisition artifacts.
        """
        add_noise = tio.RandomNoise(std=(np.random.rand() / 4))
        return add_noise(input_sub)

    def field_bias(self, input_sub):
        """
        Introduces intensity bias field artifacts.
        - MRI scans often suffer from intensity inhomogeneity (bias fields).
        - This augmentation simulates those artifacts by applying a smooth intensity distortion.
        """
        add_bias = tio.RandomBiasField(coefficients=(np.random.rand() / 2))
        return add_bias(input_sub)

    def motion(self, input_sub):
        """
        Simulates motion artifacts caused by patient movement during scanning.
        - Introduces a synthetic motion effect in the image.
        """
        return self.add_motion(input_sub)

    def apply_augmentations(self, input_sub, aug_level):
        """Apply augmentations based on augmentation level."""
        all_functions = [self.blur, self.anistropy, self.noise, self.field_bias, self.motion]
        blur_functions = [self.noise, self.field_bias, self.anistropy]
        other_functions = [self.motion, self.blur]

        if aug_level == 0:
            selected_functions = []
        elif aug_level == 1:
            selected_functions = random.sample(all_functions, 1)
        elif aug_level == 2:
            selected_functions = random.sample(blur_functions, 1) + random.sample(other_functions, 1)
        elif aug_level == 3:
            selected_functions = random.sample(blur_functions, 2) + random.sample(other_functions, 2)

        # Apply base affine and elastic deformations
        input_sub = self.affine(input_sub)
        input_sub = self.elastix(input_sub)

        # Apply selected augmentations
        for func in selected_functions:
            input_sub = func(input_sub)

        return self.rescale(self.crop_pad(input_sub))