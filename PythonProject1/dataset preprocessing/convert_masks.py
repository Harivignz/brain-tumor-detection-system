import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

# ✅ Step 1: Define Paths
dataset_path = r"D:\pycharm\BraTS2021"  # Update with your dataset path
output_path = r"D:\pycharm\mask_folder"

# ✅ Step 2: Ensure Output Folder Exists
if not os.path.exists(output_path):
    os.makedirs(output_path)

# ✅ Step 3: Resize & Convert Each Mask File
target_shape = (128, 128, 128)  # Same as preprocessed MRI scans

for patient_folder in sorted(os.listdir(dataset_path)):
    patient_path = os.path.join(dataset_path, patient_folder)

    # Look for the segmentation mask (_seg.nii)
    seg_file = os.path.join(patient_path, f"{patient_folder}_seg.nii")

    if os.path.exists(seg_file):
        # Load the segmentation mask
        seg_nifti = nib.load(seg_file)
        seg_data = seg_nifti.get_fdata()

        # Convert to binary mask (1 = tumor, 0 = background)
        seg_data = (seg_data > 0).astype(np.uint8)

        # Resize mask to match MRI dimensions
        scale_factors = np.array(target_shape) / np.array(seg_data.shape)
        seg_resized = zoom(seg_data, scale_factors, order=0)  # Use order=0 for nearest-neighbor (binary mask)

        # Save as .npy file
        np.save(os.path.join(output_path, f"{patient_folder}_mask.npy"), seg_resized)
        print(f"✅ Converted & Resized: {patient_folder}_mask.npy")
    else:
        print(f"❌ Mask not found for: {patient_folder}")

print("🎯 Mask conversion complete!")
