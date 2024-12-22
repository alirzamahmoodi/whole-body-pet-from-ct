import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import skimage.io as io
import shutil


def GiveImageAndTargetLists(main_path):
    CT_list = []
    PET_list = []

    for patient_folder in os.listdir(main_path):
        patient_path = os.path.join(main_path, patient_folder)
        if os.path.isdir(patient_path):
            for study_folder in os.listdir(patient_path):
                study_path = os.path.join(patient_path, study_folder)
                if os.path.isdir(study_path):
                    CT_path = os.path.join(study_path, "CT.nii.gz")     # updated to handle *.nii.gz instead of *.mhd
                    PET_path = os.path.join(study_path, "PET.nii.gz")   # updated to handle *.nii.gz instead of *.mhd
                    if os.path.exists(CT_path) and os.path.exists(PET_path):
                        # Load the CT and PET images
                        CT_img = nib.load(CT_path)
                        PET_img = nib.load(PET_path)
                        
                        # Get the number of slices
                        CT_slices = CT_img.shape[2]
                        PET_slices = PET_img.shape[2]
                        
                        # Ensure the number of slices are aligned and constrained to the number of PET slices
                        if CT_slices >= PET_slices:
                            CT_list.append(CT_path)
                            PET_list.append(PET_path)
                        else:
                            print(f"Skipping {study_path} due to fewer CT slices: CT {CT_slices}, PET {PET_slices}")

    return CT_list, PET_list


def SavingAsNpy(CT_list, PET_list, CT_Tr_path, PET_Tr_path, CT_Ts_path, PET_Ts_path, CT_Va_path, PET_Va_path, prefix=""):
    count_ts = 0
    count_tr = 0
    count_va = 0

    for j in range(len(CT_list)):
        print(j, '/', len(CT_list) - 1)
        key = CT_list[j].split(os.sep)[-3]  # Assuming patient folder is third from the end in the path

        # Split data into train, validation, and test
        if j < len(CT_list) * 0.1:
            # Test set (10%)
            CT_path, PET_path = CT_Ts_path, PET_Ts_path
            count = count_ts
            count_ts += 1
        elif j < len(CT_list) * 0.2:
            # Validation set (10%)
            CT_path, PET_path = CT_Va_path, PET_Va_path
            count = count_va
            count_va += 1
        else:
            # Training set (80%)
            CT_path, PET_path = CT_Tr_path, PET_Tr_path
            count = count_tr
            count_tr += 1

        # Load the NIfTI files
        CT = sitk.ReadImage(CT_list[j])
        PET = sitk.ReadImage(PET_list[j])

        # Convert the images to NumPy arrays
        CT_np = sitk.GetArrayFromImage(CT)
        PET_np = sitk.GetArrayFromImage(PET)

        # Save as .npy
        dst_CT_name = "IC_" + prefix + '_' + str(count).zfill(6) + ".npy"
        dst_CT_path = os.path.join(CT_path, dst_CT_name)
        np.save(dst_CT_path, CT_np)

        dst_PET_name = "IC_" + prefix + '_' + str(count).zfill(6) + ".npy"
        dst_PET_path = os.path.join(PET_path, dst_PET_name)
        np.save(dst_PET_path, PET_np)

    return (count_ts, count_tr, count_va)



if __name__=='__main__':
    # Destination directory
    main_folder = 'E:/npy_output/CTtoPET/'
    os.makedirs(main_folder,exist_ok=True)

    # Train
    CT_Tr_path = os.path.join(main_folder, "trainA")
    os.makedirs(CT_Tr_path,exist_ok=True)

    PET_Tr_path = os.path.join(main_folder, "trainB")
    os.makedirs(PET_Tr_path,exist_ok=True)

    # Val
    CT_Va_path = os.path.join(main_folder, "valA")
    os.makedirs(CT_Va_path,exist_ok=True)

    PET_Va_path = os.path.join(main_folder, "valB")
    os.makedirs(PET_Va_path,exist_ok=True)

    # Test
    CT_Ts_path = os.path.join(main_folder, "testA")
    os.makedirs(CT_Ts_path,exist_ok=True)

    PET_Ts_path = os.path.join(main_folder, "testB")
    os.makedirs(PET_Ts_path,exist_ok=True)

    #Getting images_list & target_list
    raw_dot_m_files = 'E:/nifti_output/FDG-PET-CT-Lesions/'
    CT_list, PET_list = GiveImageAndTargetLists(raw_dot_m_files)
    print("len(CT_list) & len(PET_list):",len(CT_list),'  &  ' ,len(PET_list))
    prefix = ""
    #Let's shuffle them
    indices = np.arange(len(CT_list))
    np.random.shuffle(indices)
    CT_list, PET_list = np.array(CT_list),  np.array(PET_list)
    CT_list = CT_list[indices]
    PET_list = PET_list[indices]
    #SavingD
    SavingAsNpy(CT_list,    PET_list,
                CT_Tr_path,  PET_Tr_path,
                CT_Ts_path,  PET_Ts_path,
                CT_Va_path,  PET_Va_path,
                prefix=prefix)
