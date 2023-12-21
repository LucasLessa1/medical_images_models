import os
import numpy as np
import imageio
import pandas as pd
from sklearn.model_selection import train_test_split


def build_dataset(folder_path):
    data = {'name': [], 'path': [], 'label': []}

    # Iterate through subdirectories (labels)
    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        
        # Check if the item is a directory
        if os.path.isdir(label_path):
            # Iterate through image files in the subdirectory
            for image in os.listdir(label_path):
                image_path = os.path.join(label_path, image)
                
                # Check if the item is a file and is an image file
                if os.path.isfile(image_path) and image.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Append image details to the dataset
                    data['name'].append(image)
                    data['path'].append(image_path)
                    data['label'].append(label)
    
    # Create a DataFrame from the collected data
    dataset = pd.DataFrame(data)
    return dataset

def make_dataset_by_folder(base_path, label_column):
    dataset = build_dataset(base_path)
    train_df, test_df, val_df = split_dataset_by_label(dataset, label_column=label_column, train_size=0.8, test_size=0.1, val_size=0.1)
    compare_label_counts(dataset, train_df, desired_proportion=0.8)
    compare_label_counts(dataset, test_df, desired_proportion=0.1)
    compare_label_counts(dataset, val_df, desired_proportion=0.1)
    
    return train_df, test_df, val_df 

def make_dataset_by_df(paths_image, paths_df, label_column):
    train_df = pd.read_csv(paths_df[0])
    train_df = add_image_paths_to_dataframe(train_df, paths_image[0], column_name='image_id')
    
    test_df = pd.read_csv(paths_df[1])
    test_df = add_image_paths_to_dataframe(test_df, paths_image[1], column_name='image_id')
    
    train_df, val_df = split_dataset_by_label(train_df, label_column=label_column, train_size=0.9, test_size=0, val_size=0.1, return_test=False)
    
    return train_df, test_df, val_df


def split_dataset_by_label(dataframe, train_size, test_size, val_size, label_column="label", return_test=True):
    # Shuffle the dataframe
    shuffled_data = dataframe.sample(frac=1, random_state=42).reset_index(drop=True)

    # Calculate the counts of each label
    label_counts = shuffled_data[label_column].value_counts()

    train_data = pd.DataFrame()
    test_data = pd.DataFrame()
    val_data = pd.DataFrame()

    # Iterate through unique labels
    for label in label_counts.index:
        label_data = shuffled_data[shuffled_data[label_column] == label]
        train, test_val = train_test_split(label_data, test_size=(1 - train_size), random_state=42)
        
        if return_test:
            test, val = train_test_split(test_val, test_size=(val_size / (val_size + test_size)), random_state=42)
            test_data = pd.concat([test_data, test])

            if val_size > 0:
                val_data = pd.concat([val_data, val])
        else:
            # Compute the size of the validation set when not returning the test set
            if val_size > 0:
                val_size_label = int(len(test_val) * (val_size / (val_size + test_size)))
                val_data = pd.concat([val_data, test_val.sample(n=val_size_label, random_state=42)])

        train_data = pd.concat([train_data, train])

    if return_test and val_size > 0:
        return train_data, test_data, val_data
    else:
        return train_data, val_data



def compare_label_counts(original_df, train_df, desired_proportion, label_column='label'):
    label_counts_original = original_df[label_column].value_counts()
    
    for label, count_original in label_counts_original.items():
        expected_count = int(len(original_df.loc[original_df[label_column] == label]) * desired_proportion)
        actual_count = len(train_df.loc[train_df[label_column] == label])

        if expected_count == actual_count:
            print(f"Is equal")
        else:
            print(f"Label '{label}' is not equal. Expected - {expected_count}, Actual - {actual_count}")
            
            
def add_image_paths_to_dataframe(dataframe, folder_path, column_name):
    image_paths = []

    # Get the list of image files in the folder
    image_files = os.listdir(folder_path)
    image_files = [f for f in image_files if f.endswith('.jpg')]

    # Create a dictionary mapping image_id to image file names in the folder
    image_id_to_file = {file.split('.')[0]: file for file in image_files}

    # Iterate through 'image_id' column in the DataFrame
    for image_id in dataframe[column_name]:
        # Check if the image_id exists in the dictionary mapping
        if image_id in image_id_to_file:
            image_file = os.path.join(folder_path, image_id_to_file[image_id])
            image_paths.append(image_file)
        else:
            image_paths.append(None)  # If image doesn't exist, insert None
    
    # Add a new column 'path' with the image paths to the DataFrame
    dataframe['path'] = image_paths
    
    return dataframe


def get_label_counts_and_print(dataframe, label_column):
    total_images = len(dataframe)  # Total number of images in the dataframe
    label_counts = dataframe[label_column].value_counts().to_dict()

    print(f"Total number of images: {total_images}")
    print(f"Number of unique labels: {len(label_counts)}")
    for label, count in label_counts.items():
        print(f"Label '{label}' has {count} images.")

    return label_counts


import imageio

def image_analysis(dataframe, path_column='path'):
    smallest_pixel = float('inf')
    largest_pixel = 0
    total_images = 0
    channel_values = {'R': [], 'G': [], 'B': []}
    is_gray = False
    # Iterate through each image in the dataframe
    for index, row in dataframe.iterrows():
        image_path = row[path_column]
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Read the image using imageio
            image = imageio.imread(image_path)

            # Check if the image is grayscale or RGB
            if len(image.shape) == 2:  # Grayscale image
                min_pixel = np.min(image)
                max_pixel = np.max(image)
                smallest_pixel = min(smallest_pixel, min_pixel)
                largest_pixel = max(largest_pixel, max_pixel)
                is_gray = True
                total_images += 1
                
                continue
                # image = np.expand_dims(image, axis=-1)  # Add a channel dimension
            elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBA image
                image = image[:, :, :3]  # Remove alpha channel

            # Check for smallest and largest pixel value
            min_pixel = np.min(image)
            max_pixel = np.max(image)
            smallest_pixel = min(smallest_pixel, min_pixel)
            largest_pixel = max(largest_pixel, max_pixel)

            # Extract channel-wise values
            channels = np.dsplit(image, image.shape[-1])
            for i, channel in enumerate(channels):
                mean_val = np.mean(channel)
                std_val = np.std(channel)
                channel_values['R' if i == 0 else 'G' if i == 1 else 'B'].append((mean_val, std_val))
            
            total_images += 1
    
    # Print the results
    print(f"Smallest pixel value: {smallest_pixel}")
    print(f"Largest pixel value: {largest_pixel}")
    print(f"Total images processed: {total_images}")
    
    channel_stats = {}
    if is_gray:
        return {
        'smallest_pixel_value': smallest_pixel,
        'largest_pixel_value': largest_pixel,
        'total_images': total_images,
        'channel_statistics': channel_stats,
        'channels': 1
    }
    else:
        # Calculate average and standard deviation per channel
        for channel, values in channel_values.items():
            avg = np.mean([val[0] for val in values])
            std_dev = np.mean([val[1] for val in values])
            channel_stats[channel] = {'average': avg, 'std_dev': std_dev}
        
        print("Channel Statistics:")
        for channel, stats in channel_stats.items():
            print(f"Channel '{channel}':")
            print(f"  - Average: {stats['average']}")
            print(f"  - Standard Deviation: {stats['std_dev']}")
        
        # Return results as a dictionary
        return {
            'smallest_pixel_value': smallest_pixel,
            'largest_pixel_value': largest_pixel,
            'total_images': total_images,
            'channel_statistics': channel_stats,
            'channels': 3
        }



def check_images_existence(dataframe, path_column='path'):
    cleaned_dataframe = dataframe.copy()
    removed_images = []

    # Iterate through each row in the DataFrame
    for index, row in dataframe.iterrows():
        image_path = row[path_column]

        if image_path == None or not os.path.exists(image_path):
            removed_images.append(row)
            cleaned_dataframe.drop(index, inplace=True)
            print(f"Image not found in folder: {image_path}")
    
    # Print removed lines
    if removed_images:
        print("Removed lines:")
        for row in removed_images:
            print(row)
    
    return cleaned_dataframe



import imageio

def analyze_image_shapes(dataframe, min_shape, path_column='path'):
    total_images = 0
    total_height = 0
    total_width = 0
    smaller_than_x_count = 0

    # Iterate through each row in the dataframe
    for index, row in dataframe.iterrows():
        image_path = row[path_column]
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Read the image using imageio
            image = imageio.imread(image_path)
            if len(image.shape) == 2:  # Check if image is grayscale
                height, width = image.shape
            else:  # Color image (RGB)
                height, width, _ = image.shape

            total_images += 1
            total_height += height
            total_width += width

            # Check if the image shape is smaller than min_shape
            if height < min_shape[0] or width < min_shape[1]:
                smaller_than_x_count += 1

    # Calculate the average shape of the images
    average_height = total_height / total_images if total_images > 0 else 0
    average_width = total_width / total_images if total_images > 0 else 0

    # Print the results
    print(f"Average image shape - Height: {average_height}, Width: {average_width}")
    print(f"Number of images with shape smaller than {min_shape}: {smaller_than_x_count}")

    # Return results as a dictionary
    return {
        'average_height': average_height,
        'average_width': average_width,
        'smaller_than_x_count': smaller_than_x_count
    }
