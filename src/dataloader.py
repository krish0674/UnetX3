import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from .misc import list_image_paths,normalize_data

import cv2
import numpy as np
import albumentations as albu
from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self, high_res_folder, low_res_folder, transform=None, augmentation=None):
        self.image_pairs = list_image_paths(high_res_folder, low_res_folder)  # Ensure this function returns pairs of paths
        self.transform = transform
        self.augmentation = augmentation

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        high_res_path, low_res_path = self.image_pairs[idx]

        # Load images
        high_res_image = cv2.imread(high_res_path, cv2.IMREAD_GRAYSCALE)
        low_res_image = cv2.imread(low_res_path, cv2.IMREAD_GRAYSCALE)

        # Add a channel dimension for Albumentations
        high_res_image = np.expand_dims(high_res_image, axis=-1)
        low_res_image = np.expand_dims(low_res_image, axis=-1)

        if self.augmentation:
            augmented = self.augmentation(image=high_res_image, image1=low_res_image)  # Use 'image1' as per additional_targets
            high_res_image, low_res_image = augmented['image'], augmented['image1']

        if self.transform:
            transformed_high_res = self.transform(image=high_res_image)
            transformed_low_res = self.transform(image=low_res_image)
            high_res_image, low_res_image = transformed_high_res['image'], transformed_low_res['image']

        # Normalize data if needed (ensure normalize_data is suitable for your data)
        high_res_image = normalize_data(high_res_image)
        low_res_image = normalize_data(low_res_image)

        burst_image = np.concatenate([low_res_image] * 10, axis=1)

        return burst_image, high_res_image
