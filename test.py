import zipfile
import os

def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

dataset_zip = '/mnt/nas/LucasLessa/COVID_Dataset_original.zip'
extract_to =  '/mnt/nas/LucasLessa/medical_images_models/COVID_Dataset_original'

print(dataset_zip)
extract_zip(dataset_zip, extract_to)
