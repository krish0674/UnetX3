import matplotlib.pyplot as plt
import os

# normalizing target image to be compatible with tanh activation function
def normalize_data(image):
    image = image.float()
    image /= 255.0
    image = 2 * image - 1
    return image

def unnormalize_data(data):
    data += 1
    data /= 2
    return data

def standardize_data(data):
    """
    Standardize the data to have zero mean and unit variance.

    Parameters:
        data (numpy.ndarray): Input data to be standardized.

    Returns:
        numpy.ndarray: Standardized data.
    """
    standardized_data = (data - np.mean(data)) / np.std(data)
    return standardized_data

# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()

def list_img(dir1):
    lst = []
    for root, dirs, files in os.walk(dir1):
        lst.extend(files)
    lst = sorted(lst)
    for x in range(len(lst)):
        lst[x]= dir1+ '/'+ lst[x]
    return lst

import os

def list_image_paths(high_res_folder, low_res_folder):
    high_res_files = sorted([f for f in os.listdir(high_res_folder) if os.path.isfile(os.path.join(high_res_folder, f))])
    
    low_res_files = sorted([f for f in os.listdir(low_res_folder) if os.path.isfile(os.path.join(low_res_folder, f))])
    
    pairs = []
    for hr_file in high_res_files:
        if hr_file in low_res_files:
            hr_path = os.path.join(high_res_folder, hr_file)
            lr_path = os.path.join(low_res_folder, hr_file)
            pairs.append((hr_path, lr_path))
    
    return pairs