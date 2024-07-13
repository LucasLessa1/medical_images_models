import zipfile
import os



def extract_zip(zip_path, extract_to='.'):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

## Dataset
# https://drive.google.com/file/d/12486kfJmFGLrVzMGGVM4kOIBOjwG5BOS/view?usp=sharing

## modules
# https://drive.google.com/file/d/1wp9rTiYwX3dJgbwdvifc86_CsC37q4hj/view?usp=sharing
# Download files

dataset_zip = '/mnt/nas/LucasLessa/COVID_Dataset_original.zip'

print(dataset_zip)
# Extract files
extract_zip(dataset_zip)
