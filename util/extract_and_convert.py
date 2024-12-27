import numpy as np
import skimage.io as io
import skimage.transform as transform
import os

def extract_and_convert(pred_path, subject_id):
    # Load the .npy files
    ct_file = os.path.join(pred_path, f"{subject_id}_CT.npy")
    original_pet_file = os.path.join(pred_path, f"{subject_id}_OriginalPET.npy")
    synthetic_pet_file = os.path.join(pred_path, f"{subject_id}_SyntheticPET.npy")

    ct_data = np.load(ct_file)
    original_pet_data = np.load(original_pet_file)
    synthetic_pet_data = np.load(synthetic_pet_file)

    # Find the coronal slice with SUVmax > 2.5 in synthetic PET data
    coronal_slices = synthetic_pet_data.max(axis=0)
    slice_index = np.argmax(coronal_slices > 2.5)

    if coronal_slices[slice_index] <= 2.5:
        print("No coronal slice with SUVmax > 2.5 found.")
        return

    # Extract the coronal slice
    ct_slice = ct_data[:, slice_index, :]
    original_pet_slice = original_pet_data[:, slice_index, :]
    synthetic_pet_slice = synthetic_pet_data[:, slice_index, :]

    # Convert to PNG and resize to 512x512 while maintaining aspect ratio
    def save_resized_slice(slice_data, folder, subject_id):
        slice_resized = transform.resize(slice_data, (512, 512), preserve_range=True, anti_aliasing=True)
        slice_resized = (slice_resized / slice_resized.max() * 255).astype(np.uint8)
        out_file = os.path.join(folder, f"{subject_id}.png")
        io.imsave(out_file, slice_resized)

    # Create directories if they don't exist
    checkpoints_dir = "checkpoints"
    ct_folder = os.path.join(checkpoints_dir, "CT")
    original_pet_folder = os.path.join(checkpoints_dir, "OriginalPET")
    synthetic_pet_folder = os.path.join(checkpoints_dir, "SyntheticPET")

    os.makedirs(ct_folder, exist_ok=True)
    os.makedirs(original_pet_folder, exist_ok=True)
    os.makedirs(synthetic_pet_folder, exist_ok=True)

    # Save the slices
    save_resized_slice(ct_slice, ct_folder, subject_id)
    save_resized_slice(original_pet_slice, original_pet_folder, subject_id)
    save_resized_slice(synthetic_pet_slice, synthetic_pet_folder, subject_id)

if __name__ == '__main__':
    pred_path = '/code/experiment_name/npy_results'
    subject_id = 'PETCT_efd619ecbf'

    extract_and_convert(pred_path, subject_id)
