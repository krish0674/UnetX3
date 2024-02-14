import albumentations as albu
import cv2  # Import cv2 to use cv2.INTER_CUBIC

def get_training_augmentation():
    train_transform = [
        # Specify bicubic interpolation using cv2.INTER_CUBIC
        albu.Resize(448, 640, interpolation=cv2.INTER_CUBIC, always_apply=True),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
    ]
    return albu.Compose(train_transform, additional_targets={'image1': 'mask'}, is_check_shapes=False)
    
def get_validation_augmentation():
    test_transform = [
        # Specify bicubic interpolation using cv2.INTER_CUBIC
        albu.Resize(448, 640, interpolation=cv2.INTER_CUBIC, always_apply=True),
    ]
    return albu.Compose(test_transform, additional_targets={'image1': 'mask'}, is_check_shapes=False)


def to_tensor(x, **kwargs):
    # If x is a 2D array (grayscale image), add a new axis to create a single channel
    if len(x.shape) == 2:
        x = x[np.newaxis, :, :].astype('float32')
    else:
        # For a 3D array (already has channels), transpose from (H, W, C) to (C, H, W)
        x = x.transpose(2, 0, 1).astype('float32')
    return x


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    _transform = [
        albu.Resize(256,256),
        #albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform, is_check_shapes=False)    