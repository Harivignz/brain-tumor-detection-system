import os
import tarfile
import zipfile
import gzip
import shutil

def extract_file(file_path, extract_path):
    """Extracts TAR, ZIP, and GZ files recursively"""
    if file_path.endswith(".tar") or file_path.endswith(".tar.gz"):
        with tarfile.open(file_path, 'r') as tar:
            tar.extractall(path=extract_path)
        os.remove(file_path)  # Delete extracted archive

    elif file_path.endswith(".zip"):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        os.remove(file_path)  # Delete extracted archive

    elif file_path.endswith(".gz"):  # Handle .nii.gz files
        extracted_file = file_path[:-3]  # Remove .gz extension
        with gzip.open(file_path, 'rb') as f_in:
            with open(extracted_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(file_path)  # Delete the original .gz file

def recursive_extract(root_path):
    """Walks through folders and extracts all nested archives"""
    for root, _, files in os.walk(root_path):
        for file in files:
            file_path = os.path.join(root, file)
            extract_file(file_path, root)  # Extract in place

# Define the correct extraction path
main_extract_path = "D:\\pycharm"

# First, extract the main tar/zip file
main_archive_path = "C:\\Users\\harip\\PycharmProjects\\PythonProject2\\BraTS2021_Training_Data.tar"  # Update if it's a ZIP
extract_file(main_archive_path, main_extract_path)

# Now recursively extract nested archives
recursive_extract(main_extract_path)

print(f"✅ Extraction complete! Check: {main_extract_path}")
