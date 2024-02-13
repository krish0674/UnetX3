import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from .misc import list_image_paths,normalize_data

import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from .misc import list_image_paths, normalize_data

class Dataset(Dataset):
    def __init__(self, high_res_folder, low_res_folder, transform=None, augmentation=None):
        self.image_pairs = list_image_paths(high_res_folder, low_res_folder)
        self.transform = transform
        self.augmentation = augmentation

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        high_res_path, low_res_path = self.image_pairs[idx]

        # Load images as grayscale
        high_res_image = cv2.imread(high_res_path, cv2.IMREAD_GRAYSCALE)
        low_res_image = cv2.imread(low_res_path, cv2.IMREAD_GRAYSCALE)

        # Ensure images are 2D arrays (H, W), add a channel dimension (H, W, 1) for Albumentations
        high_res_image = np.expand_dims(high_res_image, axis=-1)
        low_res_image = np.expand_dims(low_res_image, axis=-1)

        if self.augmentation:
            augmented = self.augmentation(image=high_res_image, mask=low_res_image)
            high_res_image = augmented['image']
            low_res_image = augmented['mask']
            
        if self.transform:
            high_res_image = self.transform(image=high_res_image)['image']
            low_res_image = self.transform(image=low_res_image)['image']
            high_res_image = normalize_data(high_res_image)
            low_res_image = normalize_data(low_res_image)


        return low_res_image, high_res_image
