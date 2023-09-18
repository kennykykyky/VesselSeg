import numpy as np
from skimage.filters import frangi
import matplotlib.pyplot as plt
from skimage import img_as_ubyte, exposure, feature
import os
import pdb
from skimage.metrics import structural_similarity as ssim

# Load the data
data = np.load('/home/kaiyu/project/VesselSeg/data/SMRA_MONAI_npy/imagesTr/001.npy')

mask = np.load('/home/kaiyu/project/VesselSeg/data/SMRA_MONAI_npy/labelsTr/001.npy')

# Create a directory to save the results
if not os.path.exists('./tmp/fangi_preprocess'):
    os.makedirs('./tmp/fangi_preprocess')


# Define the parameter ranges
alpha_range = np.linspace(0.4, 0.7, 3)
beta_range = np.linspace(0.4, 0.7, 3)
gamma_range = np.linspace(0.1, 1, 1)

alpha_range = [0.8]
beta_range = [0.8]

# Initialize the best parameters and the best score
best_params = (None, None, None)
best_score = 0

# Perform a grid search over the parameter ranges
for alpha in alpha_range:
    for beta in beta_range:
        for gamma in gamma_range:
            
                # print the current parameters
                score_mean = 0
                count = 0
                score_sum = 0
                for i in range(0, data.shape[2], 10):
                    count += 1
                    image = data[..., i]
                    mask_slice = mask[..., i]
                    # Apply the Frangi filter with the current parameters
                    filtered_image = frangi(image, alpha=alpha, beta=beta)
                    # filtered_image = frangi(image, alpha=alpha, beta=beta, gamma=gamma)
                    # filtered_image = frangi(image)

                    # Threshold the filtered image to create a binary segmentation
                    segmentation = filtered_image > 0.10

                    edges = feature.canny(mask_slice, sigma=3)

                    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
                    ax[0].imshow(image, cmap='gray')
                    ax[0].set_title('Original Slice')
                    ax[1].imshow(edges)
                    ax[1].set_title('Mask Slice')
                    ax[2].imshow(segmentation)
                    ax[2].set_title('Enhanced Slice')
                    # Save the figure
                    plt.savefig(f'./tmp/fangi_preprocess/frangi_try.png')
                    pdb.set_trace()

                    # Compute the Dice similarity coefficient
                    score = ssim(segmentation, edges)
                    score_sum += score
                
                # Compute the mean score for this parameter combination
                score_mean = score_sum / count
                
                # print the current parameters and the mean score
                print(f'alpha={alpha}, beta={beta}, gamma={gamma}, score={score_mean}')

                # If this score is better than the current best score, update the best score and best parameters
                if score_mean > best_score:
                    best_score = score_mean
                    best_params = (alpha, beta, gamma)

# Print the best parameters and the best score
print(f'Best parameters: alpha={best_params[0]}, beta={best_params[1]}, gamma={best_params[2]}')
print(f'Best score: {best_score}')

pdb.set_trace()

# Apply the Frangi filter to each slice and save the results
# show every 10th slice
for i in range(0, data.shape[2], 10):
    image = data[...,i]
    mask_slice = mask[...,i]
    enhanced_slice = frangi(image, alpha=0.5, beta=0.5, gamma=0.5)
    enhanced_slice = exposure.rescale_intensity(enhanced_slice, in_range='image', out_range=(0,255))
    # enhanced_slice = img_as_ubyte(enhanced_slice)
    # Compare the original and enhanced slices
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original Slice')
    ax[1].imshow(mask_slice)
    ax[1].set_title('Mask Slice')
    ax[2].imshow(enhanced_slice * 10, cmap='gray')
    ax[2].set_title('Enhanced Slice')

    # Save the figure
    plt.savefig(f'./tmp/fangi_preprocess/comparison_{i}.png')
