import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import skimage.io as io
import shutil

def resample_image(image, target_slices):
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()
    new_spacing = (
        original_spacing[0],
        original_spacing[1],
        original_spacing[2] * (original_size[2] / target_slices)
    )
    new_size = (
        original_size[0],
        original_size[1],
        target_slices
    )
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetInterpolator(sitk.sitkLinear)
    resampled_image = resample.Execute(image)
    return resampled_image

def GiveImageAndTargetLists(main_path, target_slices=280):
    CT_list = []
    PET_list = []

    for patient_folder in os.listdir(main_path):
        patient_path = os.path.join(main_path, patient_folder)
        if os.path.isdir(patient_path):
            for study_folder in os.listdir(patient_path):
                study_path = os.path.join(patient_path, study_folder)
                if os.path.isdir(study_path):
                    CT_path = os.path.join(study_path, "CT.nii.gz")
                    PET_path = os.path.join(study_path, "PET.nii.gz")
                    if os.path.exists(CT_path) and os.path.exists(PET_path):
                        # Load the CT and PET images
                        CT_img = sitk.ReadImage(CT_path)
                        PET_img = sitk.ReadImage(PET_path)

                        # Resample the images to the target number of slices
                        CT_img_resampled = resample_image(CT_img, target_slices)
                        PET_img_resampled = resample_image(PET_img, target_slices)

                        # Convert to numpy arrays
                        CT_np = sitk.GetArrayFromImage(CT_img_resampled)
                        PET_np = sitk.GetArrayFromImage(PET_img_resampled)

                        # Ensure the correct number of channels (7)
                        CT_np_stacks = [CT_np[i:i+7, :, :] for i in range(0, CT_np.shape[0], 7)]
                        PET_np_stacks = [PET_np[i:i+7, :, :] for i in range(0, PET_np.shape[0], 7)]

                        for i, (ct_stack, pet_stack) in enumerate(zip(CT_np_stacks, PET_np_stacks)):
                            if ct_stack.shape[0] < 7:
                                ct_stack = np.pad(ct_stack, ((0, 7 - ct_stack.shape[0]), (0, 0), (0, 0)), mode='constant')
                            if pet_stack.shape[0] < 7:
                                pet_stack = np.pad(pet_stack, ((0, 7 - pet_stack.shape[0]), (0, 0), (0, 0)), mode='constant')

                            # Save the resampled images to temporary files
                            CT_resampled_path = os.path.join(study_path, f"CT_resampled_{i}.nii.gz")
                            PET_resampled_path = os.path.join(study_path, f"PET_resampled_{i}.nii.gz")
                            sitk.WriteImage(sitk.GetImageFromArray(ct_stack), CT_resampled_path)
                            sitk.WriteImage(sitk.GetImageFromArray(pet_stack), PET_resampled_path)

                            CT_list.append(CT_resampled_path)
                            PET_list.append(PET_resampled_path)

    return CT_list, PET_list

def SavingAsNpy(CT_list, PET_list, CT_Tr_path, PET_Tr_path, CT_Ts_path, PET_Ts_path, CT_Va_path, PET_Va_path, prefix=""):
    count_ts = 0
    count_tr = 0
    count_va = 0

    for j in range(len(CT_list)):
        print(j, '/', len(CT_list) - 1)
        key = CT_list[j].split(os.sep)[-3]

        # Split data into train, validation, and test
        if j < len(CT_list) * 0.1:
            CT_path, PET_path = CT_Ts_path, PET_Ts_path
            count = count_ts
            count_ts += 1
        elif j < len(CT_list) * 0.2:
            CT_path, PET_path = CT_Va_path, PET_Va_path
            count = count_va
            count_va += 1
        else:
            CT_path, PET_path = CT_Tr_path, PET_Tr_path
            count = count_tr
            count_tr += 1

        # Load and resample the NIfTI files
        CT = sitk.ReadImage(CT_list[j])
        PET = sitk.ReadImage(PET_list[j])
        CT_resampled = resample_image(CT, target_slices=280)
        PET_resampled = resample_image(PET, target_slices=280)

        # Convert to NumPy arrays
        CT_np = sitk.GetArrayFromImage(CT_resampled)
        PET_np = sitk.GetArrayFromImage(PET_resampled)

        # Save as .npy
        dst_CT_name = "IC_" + prefix + '_' + str(count).zfill(6) + ".npy"
        dst_CT_path = os.path.join(CT_path, dst_CT_name)
        np.save(dst_CT_path, CT_np)

        dst_PET_name = "IC_" + prefix + '_' + str(count).zfill(6) + ".npy"
        dst_PET_path = os.path.join(PET_path, dst_PET_name)
        np.save(dst_PET_path, PET_np)

    return (count_ts, count_tr, count_va)


if __name__ == '__main__':
    # Destination directories
    main_folder = 'E:/npy_output/CTtoPET/'
    os.makedirs(main_folder, exist_ok=True)

    # Train
    CT_Tr_path = os.path.join(main_folder, "trainA")
    os.makedirs(CT_Tr_path, exist_ok=True)
    PET_Tr_path = os.path.join(main_folder, "trainB")
    os.makedirs(PET_Tr_path, exist_ok=True)

    # Val
    CT_Va_path = os.path.join(main_folder, "valA")
    os.makedirs(CT_Va_path, exist_ok=True)
    PET_Va_path = os.path.join(main_folder, "valB")
    os.makedirs(PET_Va_path, exist_ok=True)

    # Test
    CT_Ts_path = os.path.join(main_folder, "testA")
    os.makedirs(CT_Ts_path, exist_ok=True)
    PET_Ts_path = os.path.join(main_folder, "testB")
    os.makedirs(PET_Ts_path, exist_ok=True)

    # Get images list & target list
    raw_dot_m_files = 'FDG-PET-CT-Lesions'
    CT_list, PET_list = GiveImageAndTargetLists(raw_dot_m_files)
    print("len(CT_list) & len(PET_list):", len(CT_list), ' & ', len(PET_list))
    prefix = ""

    # Shuffle
    indices = np.arange(len(CT_list))
    np.random.shuffle(indices)
    CT_list, PET_list = np.array(CT_list), np.array(PET_list)
    CT_list = CT_list[indices]
    PET_list = PET_list[indices]

    # Saving data
    SavingAsNpy(CT_list, PET_list, CT_Tr_path, PET_Tr_path, CT_Ts_path, PET_Ts_path, CT_Va_path, PET_Va_path, prefix=prefix)
