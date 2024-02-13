import wandb
import segmentation_models_pytorch as smp
from .train_utils import TrainEpoch, ValidEpoch
from .loss import custom_loss, custom_lossv,lossX3
from .dataloader import Dataset #,list_image_paths
from .transformations import get_training_augmentation, get_validation_augmentation, get_preprocessing
from .model import UnetX3
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics import PeakSignalNoiseRatio
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
#from torch.utils.data import Dataset, DataLoader

def train(epochs, batch_size, hr_dir, tar_dir, hr_val_dir, tar_val_dir, encoder='resnet34', encoder_weights='imagenet', device='cuda', lr=1e-4):
    activation = 'tanh' 
    # create segmentastion model with pretrained encoder
    model = UnetX3(
        activation=activation,
        encoder_name=encoder, 
        encoder_weights=encoder_weights, 
    )

    def get_transform():
        return A.Compose([
            ToTensorV2(),
    ])

    transform = get_transform()
    
    train_dataset = Dataset(
    high_res_folder=tar_dir,
    low_res_folder=hr_dir,
    augmentation=get_training_augmentation(),  
    transform=transform,
    )


    val_dataset = Dataset(
    high_res_folder=tar_val_dir,
    low_res_folder=hr_val_dir,
    augmentation=get_validation_augmentation(),  
    transform=transform
    )

    # Create the DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    

    # loss = custom_loss(batch_size, beta=beta)
    # lossV = custom_lossv()
    
    # image_paths, mask_paths = list_image_paths(hr_dir)

    # train_dataset = Dataset(image_paths, mask_paths, transform=transform)

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # image_paths1, mask_paths1 = list_image_paths(hr_val_dir)

    # val_dataset = Dataset(image_paths1, mask_paths1, transform=transform)

    # valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    loss=lossX3()

    Z = StructuralSimilarityIndexMeasure()
    P = PeakSignalNoiseRatio()
    P.__name__ = 'psnr'
    Z.__name__ = 'ssim'
    metrics = [
        Z,
        P,
    ]

    optimizer = torch.optim.Adam([ 
        dict(params=model.parameters(), lr=lr),
    ])
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,250)
    train_epoch = TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=device,
        verbose=True,
    )
    valid_epoch = ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=device,
        verbose=True,
    )

    max_ssim = 0
    max_psnr = 0
    counter = 0
    for i in range(0, epochs):
        
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        # scheduler.step()
        print(train_logs)
        wandb.log({'epoch':i+1,'t_loss':train_logs['loss_x3'],'t_ssim':train_logs['ssim'],'v_loss':valid_logs['loss_x3'],'v_ssim':valid_logs['ssim'],'v_psnr':valid_logs['psnr'],'t_psnr':train_logs['psnr']})
        if max_ssim <= valid_logs['ssim']:
            max_ssim = valid_logs['ssim']
            max_psnr = valid_logs['psnr']
            wandb.config.update({'max_ssim':max_ssim, 'max_psnr':max_psnr}, allow_val_change=True)
            torch.save(model.state_dict(), './best_model.pth')
            print('Model saved!')
            counter = 0
        counter = counter+1
    print(f'max ssim: {max_ssim} max psnr: {max_psnr}')

def train_model(configs):
    train(configs['epochs'], configs['batch_size'], configs['hr_dir'], configs['tar_dir'],
         configs['hr_val_dir'], configs['tar_val_dir'], configs['encoder'],
         configs['encoder_weights'], configs['device'], configs['lr'])
         