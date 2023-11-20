import gzip
import nibabel as nib
import os

gzipped_nii_file_train = 'Task07_Pancreas/imagesTr'
output_directory = 'Task07_Pancreas/data/imagesTr'  # save path

os.chdir(gzipped_nii_file_train)
for file in os.listdir(gzipped_nii_file_train):
    if not file.startswith("."):
        file_path = os.path.join(gzipped_nii_file_train, file)
        with gzip.open(file_path, 'rb') as gz_file:
            output_file_name = file.replace('.gz', '')
            output_file_path = os.path.join(output_directory, output_file_name)
            
            with open(output_file_path, 'wb') as output_file:
                output_file.write(gz_file.read())
                
            nii_data = nib.load(output_file_path)
            image_data = nii_data.get_fdata()

print("imagesTr unzipped")

gzipped_nii_file_test = 'Task07_Pancreas/imagesTs'
output_directory = 'Task07_Pancreas/data/imagesTs'  # save path

os.chdir(gzipped_nii_file_test)
for file in os.listdir(gzipped_nii_file_test):
    if not file.startswith("."):
        file_path = os.path.join(gzipped_nii_file_test, file)
        with gzip.open(file_path, 'rb') as gz_file:
            output_file_name = file.replace('.gz', '')
            output_file_path = os.path.join(output_directory, output_file_name)
            
            with open(output_file_path, 'wb') as output_file:
                output_file.write(gz_file.read())
                
            nii_data = nib.load(output_file_path)
            image_data = nii_data.get_fdata()

print("imagesTs unzipped")

gzipped_nii_file_labels = 'Task07_Pancreas/labelsTr'
output_directory = 'Task07_Pancreas/data/labelsTr'  # save path

os.chdir(gzipped_nii_file_labels)
for file in os.listdir(gzipped_nii_file_labels):
    if not file.startswith("."):
        file_path = os.path.join(gzipped_nii_file_labels, file)
        with gzip.open(file_path, 'rb') as gz_file:
            output_file_name = file.replace('.gz', '')
            output_file_path = os.path.join(output_directory, output_file_name)
            
            with open(output_file_path, 'wb') as output_file:
                output_file.write(gz_file.read())
                
            nii_data = nib.load(output_file_path)
            image_data = nii_data.get_fdata()

print("labelsTr unzipped")