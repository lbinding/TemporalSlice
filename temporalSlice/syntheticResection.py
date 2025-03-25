
#%%
import numpy as np
import random
from scipy.ndimage import gaussian_filter, map_coordinates, binary_dilation, binary_erosion

class SyntheticResection:
    def __init__(self, T1_input, GIF_input, alpha=2, sigma=8):
        self.alpha = alpha
        self.sigma = sigma

        self.T1_input = T1_input
        self.GIF_input = GIF_input

        # Ventricle parcel values
        self.ventricles = [52, 53, 4]

        # Randomly pick between 0 or 1: 1 == Left, 0 == Right 
        left_right_choice = np.random.choice([0, 1])
        
        if left_right_choice == 1:
            self.temporal_GM = [134, 156, 186, 202, 204, 208, 172, 49, 33, 118, 124]
            self.temporal_WM = [89]
            # Create mask 
            self.gm_mask = np.isin(self.GIF_input, self.temporal_GM)
        else:
            self.temporal_GM = [133, 155, 185, 201, 203, 207, 171, 48, 32, 117, 123]
            self.temporal_WM = [81]
            # Create mask 
            self.gm_mask = np.isin(self.GIF_input, self.temporal_GM)

    def _generate_deformation(self, shape, mask):
        displacement = [np.zeros(shape, dtype=np.float32) for _ in range(len(shape))]
        for i in range(len(shape)):
            noise = np.random.uniform(-1, 1, size=shape) * mask
            smoothed_noise = gaussian_filter(noise, sigma=self.sigma, mode="constant") * self.alpha
            displacement[i] = smoothed_noise
        return displacement
    
    @staticmethod
    def _apply_deformation(image, displacement):
        coords = np.meshgrid(*[np.arange(s) for s in image.shape], indexing="ij")
        coords = [coords[i] + displacement[i] for i in range(len(image.shape))]
        return map_coordinates(image, coords, order=3, mode="nearest")
    
    @staticmethod
    def _get_coordinates(mask, max_attempts=5):
        for _ in range(max_attempts):
            coords = np.argwhere(mask)
            if coords.size > 0:
                return coords
        return None
    
    def _generate_mask(self):        
        # Get coordinates of voxels within the GM mask
        gm_voxel_coords = np.argwhere(self.gm_mask)
        
        # Randomly select a voxel
        x, y, z = random.choice(gm_voxel_coords)
        
        # Perform random walk (Growth)
        growth_steps = random.randint(5000, 100000)  # Number of growth steps
        voxels_per_step = 5  # Number of voxels to expand per step
        
        # List to store grown voxels and visited voxels
        grown_voxels = set([(x, y, z)])
        visited_voxels = set([(x, y, z)])
        
        for _ in range(growth_steps):
            selected_voxels = random.sample(list(grown_voxels), min(voxels_per_step, len(grown_voxels)))
            for x, y, z in selected_voxels:
                # Random direction
                dx, dy, dz = random.choice([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)])
                
                # Compute new voxel position
                new_x, new_y, new_z = x + dx, y + dy, z + dz
                
                # Check if within bounds, in GM, and not already visited
                if (0 <= new_x < self.GIF_input.shape[0] and
                    0 <= new_y < self.GIF_input.shape[1] and
                    0 <= new_z < self.GIF_input.shape[2] and
                    self.GIF_input[new_x, new_y, new_z] in [self.temporal_GM + self.temporal_WM][0] and
                    (new_x, new_y, new_z) not in visited_voxels):
                    
                    grown_voxels.add((new_x, new_y, new_z))
                    visited_voxels.add((new_x, new_y, new_z))
        
        
        self.resection_visited_voxels = visited_voxels
        # Create the grown mask directly using numpy instead of looping
        grown_mask = np.zeros_like(self.GIF_input)
        grown_mask[tuple(zip(*grown_voxels))] = 1  # Efficiently assign the mask
        
        # Apply Gaussian smoothing to the grown mask only
        smoothed_mask = gaussian_filter(grown_mask.astype(float), sigma=2)
        smoothed_mask = np.where(smoothed_mask >= 0.5, 1, 0)  # Threshold for inclusion

        return smoothed_mask

    def _add_demylination(self, smoothed_mask):
        dilated_mask = binary_dilation(smoothed_mask, iterations=1)
        edge_mask = np.logical_xor(smoothed_mask >0, dilated_mask)
        edge_mask = np.logical_and(self.GIF_input==self.temporal_WM, edge_mask)
        
        #Function to try 5 times to get coordinates 
        smoothed_mask_coords = self._get_coordinates(edge_mask)
        if smoothed_mask_coords is not None:
            x, y, z = random.choice(smoothed_mask_coords)
            # Perform random walk (Growth)
            growth_steps = random.randint(1000, 10000)  # Number of growth steps
            voxels_per_step = 5  # Number of voxels to expand per step
            # List to store grown voxels and visited voxels
            demylination_voxels = set([(x, y, z)])
            demylination_visited_voxels = set([(x, y, z)])
            for _ in range(growth_steps):
                selected_voxels = random.sample(list(demylination_voxels), min(voxels_per_step, len(demylination_voxels)))
                for x, y, z in selected_voxels:
                    # Random direction
                    dx, dy, dz = random.choice([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)])
                    
                    # Compute new voxel position
                    new_x, new_y, new_z = x + dx, y + dy, z + dz
                    
                    # Check if within bounds, in GM, and not already visited
                    if (0 <= new_x < self.GIF_input.shape[0] and
                        0 <= new_y < self.GIF_input.shape[1] and
                        0 <= new_z < self.GIF_input.shape[2] and
                        self.GIF_input[new_x, new_y, new_z] in self.temporal_WM and
                        (new_x, new_y, new_z) not in demylination_visited_voxels and
                        (new_x, new_y, new_z) not in self.resection_visited_voxels):
                        
                        demylination_voxels.add((new_x, new_y, new_z))
                        demylination_visited_voxels.add((new_x, new_y, new_z))
            
            # Create the grown mask directly using numpy instead of looping
            demylination_mask = np.zeros_like(self.GIF_input)
            demylination_mask[tuple(zip(*demylination_voxels))] = 1  # Efficiently assign the mask
            
            # Apply Gaussian smoothing to the grown mask only
            demylination_mask = gaussian_filter(demylination_mask.astype(float), sigma=1)
            demylination_mask = np.where(demylination_mask >= 0.1, 1, 0)  # Threshold for inclusion

            return demylination_mask
        else:
            return np.zeros_like(smoothed_mask)

    def _surrounding_deformation(self, t1_data_rm, smoothed_filled_mask, shrinked_mask, smoothed_mask):
        dilated_mask = binary_dilation(smoothed_filled_mask, iterations=10)
        edge_mask = np.logical_xor(shrinked_mask, dilated_mask)

        # Generate a deformation field only within the edge mask
        displacement_field = self._generate_deformation(t1_data_rm.shape, edge_mask)
        
        # Apply the deformation
        deformed_t1 = self._apply_deformation(t1_data_rm, displacement_field)
        deformed_smoothed_mask = self._apply_deformation(smoothed_mask, displacement_field)

        return deformed_t1, deformed_smoothed_mask
    def _resect_T1_image(self, smoothed_mask, demylination_mask):
        # Ventricles mask creation and filtering
        ventricle_mask = np.isin(self.GIF_input, self.ventricles)
        ventricle_values = self.T1_input[ventricle_mask]
        ventricle_values_filtered = ventricle_values[
            (ventricle_values > np.percentile(ventricle_values, 0)) &
            (ventricle_values < np.percentile(ventricle_values, 30))
        ]
        
        # Efficiently sample ventricle values
        num_samples = 100
        max_distance = 5
        current_value = random.choice(ventricle_values_filtered)
        samples = [current_value]
        
        for _ in range(num_samples - 1):
            nearby_values = ventricle_values_filtered[
                (ventricle_values_filtered >= current_value - max_distance) &
                (ventricle_values_filtered <= current_value + max_distance)
            ]
            next_value = random.choice(nearby_values)
            samples.append(next_value)
            current_value = next_value
        
        samples = np.array(samples)
        min_value, max_value = np.min(samples), np.max(samples)
        
        # Interpolate sampled values
        num_mask_voxels = np.sum(smoothed_mask > 0)
        interpolated_values = np.linspace(min_value, max_value, num_mask_voxels)
        np.random.shuffle(interpolated_values)
        
        # Add noise to the interpolated values
        noise_level = 1
        noise = np.random.normal(0, noise_level, size=interpolated_values.shape)
        noisy_interpolated_values = np.clip(interpolated_values + noise, min_value, max_value)
        
        # Assign noisy interpolated values to the smoothed mask
        filled_smoothed_mask = smoothed_mask.copy()
        mask_indices = np.array(np.nonzero(filled_smoothed_mask > 0)).T
        filled_smoothed_mask[tuple(mask_indices.T)] = noisy_interpolated_values
        
        # Apply further smoothing
        smoothed_filled_mask = gaussian_filter(filled_smoothed_mask.astype(float), sigma=5)
        smoothed_filled_mask = smoothed_filled_mask * (filled_smoothed_mask > 0)
        
        # Create final resection image
        t1_data_rm = self.T1_input.copy()
        t1_data_rm[smoothed_filled_mask > 1] = smoothed_filled_mask[smoothed_filled_mask > 1]
        
        # Apply Gaussian filter to the edge mask
        sigma = 0.5  # Adjust for edge smoothing strength
        smoothed_t1 = gaussian_filter(t1_data_rm.astype(float), sigma=sigma)
        
        # Edge mask creation using binary operations
        shrinked_mask = binary_erosion(smoothed_filled_mask, iterations=1)
        dilated_mask = binary_dilation(smoothed_filled_mask, iterations=1)
        edge_mask = np.logical_xor(shrinked_mask, dilated_mask)
        
        # Apply smoothing to the edge region
        t1_data_rm[edge_mask] = smoothed_t1[edge_mask]
        
        # Create demylination
        if np.any(demylination_mask):
            t1_data_rm[demylination_mask > 0] /= gaussian_filter(np.random.uniform(1.5, 2.5, size=t1_data_rm[demylination_mask > 0].shape), sigma=2)

        #Apply deformations around the T1 image
        deformed_t1,            \
        deformed_smoothed_mask  = self._surrounding_deformation(t1_data_rm, 
                                                               smoothed_filled_mask, 
                                                               shrinked_mask, 
                                                               smoothed_mask)

        return deformed_t1, deformed_smoothed_mask
    
    def create_resection(self):
        # Generates a random resection mask
        smoothed_mask = self._generate_mask()
        # Generates a random amount of demylination
        demylination_mask = self._add_demylination(smoothed_mask)
        # Application of the above to the T1 image
        deformed_t1, deformed_smoothed_mask = self._resect_T1_image(smoothed_mask, demylination_mask)

        return deformed_t1, deformed_smoothed_mask
