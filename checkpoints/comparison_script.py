from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import os
import csv

def calculate_rmse(image1, image2):
    """Calculate RMSE between two images."""
    return np.sqrt(np.mean((image1 - image2) ** 2))

def match_and_process_images(ct_folder, orgpet_folder, pixpet_folder, output_file, csv_file):
    """
    Match images from three folders based on the first 10 characters of filenames,
    calculate SSIM and RMSE for PET pairs, create a comparison table, and save metrics to CSV.

    Parameters:
        ct_folder (str): Path to the folder containing CT images.
        orgpet_folder (str): Path to the folder containing ground-truth PET images.
        pixpet_folder (str): Path to the folder containing predicted PET images.
        output_file (str): Path to save the output comparison table as a PNG file.
        csv_file (str): Path to save the SSIM and RMSE metrics as a CSV file.
    """
    # Read files and match based on the first 16 characters
    ct_files = {file[:16]: os.path.join(ct_folder, file) for file in os.listdir(ct_folder) if file.lower().endswith('.png')}
    orgpet_files = {file[:16]: os.path.join(orgpet_folder, file) for file in os.listdir(orgpet_folder) if file.lower().endswith('.png')}
    pixpet_files = {file[:16]: os.path.join(pixpet_folder, file) for file in os.listdir(pixpet_folder) if file.lower().endswith('.png')}

    matched_keys = set(ct_files.keys()) & set(orgpet_files.keys()) & set(pixpet_files.keys())
    matched_keys = sorted(matched_keys)  # Sort for consistent order

    data = []

    for key in matched_keys:
        ct_image = np.array(Image.open(ct_files[key]))
        orgpet_image = np.array(Image.open(orgpet_files[key]))
        pixpet_image = np.array(Image.open(pixpet_files[key]))

        ssim_value = ssim(orgpet_image, pixpet_image)
        rmse_value = calculate_rmse(orgpet_image, pixpet_image)

        data.append([key, ssim_value, rmse_value])

    # Save metrics to CSV
    with open(csv_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Filename', 'SSIM', 'RMSE'])
        csvwriter.writerows(data)

    # Create comparison table
    fig, ax = plt.subplots(len(matched_keys), 3, figsize=(15, 5 * len(matched_keys)))
    for i, key in enumerate(matched_keys):
        ct_image = np.array(Image.open(ct_files[key]))
        orgpet_image = np.array(Image.open(orgpet_files[key]))
        pixpet_image = np.array(Image.open(pixpet_files[key]))

        ax[i, 0].imshow(ct_image, cmap='gray')
        ax[i, 0].set_title(f'CT: {key}')
        ax[i, 0].axis('off')

        ax[i, 1].imshow(orgpet_image, cmap='gray')
        ax[i, 1].set_title(f'Org PET: {key}\nSSIM: {data[i][1]:.4f}, RMSE: {data[i][2]:.4f}')
        ax[i, 1].axis('off')

        ax[i, 2].imshow(pixpet_image, cmap='gray')
        ax[i, 2].set_title(f'Pix PET: {key}')
        ax[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

# Example usage
match_and_process_images(
    ct_folder='CT',
    orgpet_folder='OriginalPET',
    pixpet_folder='SyntheticPET',
    output_file='comparison_metrics.png',
    csv_file='comparison_metrics.csv'
)