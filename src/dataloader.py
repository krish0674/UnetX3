from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import cv2
from .misc import normalize_data, list_img

class Dataset(BaseDataset):  
    def __init__(
            self, 
            hr_dir: str, 
            #thermal_dir:str,
            tar_dir: str, 
            augmentation=None, 
            preprocessing=None,
            transform=None
    ):
        self.hr_list = list_img(hr_dir)
        #self.thermal_list= list_img(thermal_dir)
        self.tar_list = list_img(tar_dir)
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.transform=transform

    def __getitem__(self, i):
        
        # read data
        himage = cv2.imread(self.hr_list[i])
        # himage = cv2.cvtColor(himage, cv2.COLOR_BGR2RGB)
        target = cv2.imread(self.tar_list[i], 0)
        #timage = cv2.imread(self.thermal_list[i])
        # apply augmentations
        # if self.augmentation:
        #     sample = self.augmentation(image=himage,mask=target)
        #     himage, target= sample['image'], sample['mask']
        # target = target.reshape(480,640,1)
#         timage = timage.reshape(480,640,1)
        if self.transform:
            #sample = self.preprocessing(image=himage, mask=target)
            #himage, target = sample['image'], sample['mask']
           # sample = self.preprocessing(image=timage)
           # timage= sample['image']
           # sample = self.preprocessing
            
            himage = self.transform(himage)
            target = self.transform(target)
            target = target/255
            target = normalize_data(target)

        return himage,target #target#, label
        
    def __len__(self):
        return len(self.hr_list)

# import os

# def list_image_paths(main_folder):
#     image_paths = []
#     mask_paths = []

#     for subdir, dirs, files in os.walk(main_folder):
#         for file in files:
#             if file.startswith('LR'):
#                 image_path = os.path.join(subdir, file)
#                 mask_path = os.path.join(subdir, file.replace('LR', 'QM'))
#                 if os.path.exists(mask_path):
#                     image_paths.append(image_path)
#                     mask_paths.append(mask_path)

#     return image_paths, mask_paths

# from torch.utils.data import Dataset, DataLoader
# from PIL import Image

# class Dataset(Dataset):
#     def __init__(self, image_paths, mask_paths, transform=None):
#         self.image_paths = image_paths
#         self.mask_paths = mask_paths
#         self.transform = transform

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, index):
#         image_path = self.image_paths[index]
#         mask_path = self.mask_paths[index]

#         image = Image.open(image_path).convert("RGB")
#         mask = Image.open(mask_path).convert("RGB") 

#         if self.transform is not None:
#             image = self.transform(image)
#             mask = self.transform(mask)

#         return image, mask