import os
import numpy as np
import imageio
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Union, List


def build_dataset(folder_path: str) -> pd.DataFrame:
    """
    Build a dataset from images stored in subdirectories.

    Args:
    - folder_path (str): Path to the folder containing
    subdirectories with images.

    Returns:
    - dataset (pd.DataFrame): DataFrame containing image
    details (name, path, label).
    """
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
                if (os.path.isfile(image_path) and
                   image.lower().endswith(('.png', '.jpg', '.jpeg'))):
                    # Append image details to the dataset
                    data['name'].append(image)
                    data['path'].append(image_path)
                    data['label'].append(label)

    # Create a DataFrame from the collected data
    dataset = pd.DataFrame(data)
    return dataset


def make_dataset_by_folder(base_path: str,
                           label_column: str
                           ) -> Union[
                            pd.DataFrame,
                            pd.DataFrame,
                            pd.DataFrame]:
    """
    Create a dataset by splitting images into training, testing,
    and validation sets based on folders.

    Args:
    - base_path (str): Path to the base folder containing subdirectories
    with images.
    - label_column (str): Name of the column containing labels in
    the dataset.

    Returns:
    - Tuple of DataFrames: Three DataFrames representing the training,
    testing, and validation sets.
    """
    dataset = build_dataset(base_path)
    train_df, test_df, val_df = split_dataset_by_label(
        dataset,
        label_column=label_column,
        train_size=0.8,
        test_size=0.1,
        val_size=0.1)
    compare_label_counts(dataset,
                         train_df,
                         desired_proportion=0.8)
    compare_label_counts(dataset,
                         test_df,
                         desired_proportion=0.1)
    compare_label_counts(dataset,
                         val_df,
                         desired_proportion=0.1)

    return train_df, test_df, val_df


def make_dataset_by_df(paths_image: Tuple[str, str],
                       paths_df: Tuple[str, str],
                       label_column: str
                       ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create a dataset by combining image paths with DataFrame paths.

    Args:
    - paths_image (Tuple[str, str]): Tuple containing paths
    to image folders for training and testing.
    - paths_df (Tuple[str, str]): Tuple containing paths
    to DataFrames for training and testing.
    - label_column (str): Name of the column containing
    labels in the dataset.

    Returns:
    - Tuple of DataFrames: Three DataFrames representing the training,
    testing, and validation sets.
    """
    train_df = pd.read_csv(paths_df[0])
    train_df = add_image_paths_to_dataframe(train_df,
                                            paths_image[0],
                                            column_name='image_id')

    test_df = pd.read_csv(paths_df[1])
    test_df = add_image_paths_to_dataframe(test_df,
                                           paths_image[1],
                                           column_name='image_id')

    train_df, val_df = split_dataset_by_label(train_df,
                                              label_column=label_column,
                                              train_size=0.9,
                                              test_size=0,
                                              val_size=0.1,
                                              return_test=False)

    return train_df, test_df, val_df


def split_dataset_by_label(dataframe: pd.DataFrame,
                           train_size: float,
                           test_size: float,
                           val_size: float,
                           label_column: str = "label",
                           return_test: bool = True
                           ) -> Union[
                               pd.DataFrame,
                               pd.DataFrame,
                               pd.DataFrame]:
    """
    Split the dataset into training, testing, and validation sets
    based on labels.

    Args:
    - dataframe (pd.DataFrame): DataFrame containing the dataset.
    - train_size (float): Proportion of the dataset to include
    in the training set.
    - test_size (float): Proportion of the dataset to include
    in the testing set.
    - val_size (float): Proportion of the dataset to include
    in the validation set.
    - label_column (str, optional): Name of the column containing
    labels in the dataset (default: "label").
    - return_test (bool, optional): Whether to return the testing
    set (default: True).

    Returns:
    - Union of DataFrames: Three DataFrames representing the training,
    testing, and validation sets.
    """
    # Shuffle the dataframe
    shuffled_data = (dataframe.sample(frac=1, random_state=42)
                     .reset_index(drop=True))

    # Calculate the counts of each label
    label_counts = shuffled_data[label_column].value_counts()

    train_data = pd.DataFrame()
    test_data = pd.DataFrame()
    val_data = pd.DataFrame()

    # Iterate through unique labels
    for label in label_counts.index:
        label_data = shuffled_data[shuffled_data[label_column] == label]
        train, test_val = train_test_split(label_data,
                                           test_size=(1 - train_size),
                                           random_state=42)

        if return_test:
            test, val = train_test_split(test_val,
                                         test_size=(val_size /
                                                    (val_size + test_size)),
                                         random_state=42)
            test_data = pd.concat([test_data, test])

            if val_size > 0:
                val_data = pd.concat([val_data, val])
        else:

            if val_size > 0:
                val_size_label = int(len(test_val) *
                                     (val_size / (val_size + test_size)))
                val_data = pd.concat([val_data,
                                      test_val.sample(n=val_size_label,
                                                      random_state=42)])

        train_data = pd.concat([train_data, train])

    if return_test and val_size > 0:
        return train_data, test_data, val_data

    return train_data, val_data


def compare_label_counts(original_df: pd.DataFrame,
                         train_df: pd.DataFrame,
                         desired_proportion: float,
                         label_column: str = 'label') -> None:
    """
    Compare label counts between original and training datasets.

    Args:
    - original_df (pd.DataFrame): Original DataFrame
    containing the dataset.
    - train_df (pd.DataFrame): DataFrame representing
    the training set.
    - desired_proportion (float): Desired proportion
    of each label in the training set.
    - label_column (str, optional): Name of the column
    containing labels in the dataset (default: 'label').

    Returns:
    - None
    """
    label_counts_original = original_df[label_column].value_counts()

    for label, _ in label_counts_original.items():
        expected_count = int(len(original_df.loc[
            original_df[label_column] == label]) * desired_proportion)
        actual_count = len(train_df.loc[train_df[label_column] == label])

        if expected_count == actual_count:
            print("Is equal")
        else:
            print(f"Label '{label}' is not equal. Expected"
                  f" - {expected_count}, Actual - {actual_count}")


def add_image_paths_to_dataframe(dataframe: pd.DataFrame,
                                 folder_path: str,
                                 column_name: str
                                 ) -> pd.DataFrame:
    """
    Add image paths to a DataFrame based on image IDs and a folder path.

    Args:
    - dataframe (pd.DataFrame): DataFrame containing image IDs.
    - folder_path (str): Path to the folder containing the images.
    - column_name (str): Name of the column containing image IDs
    in the DataFrame.

    Returns:
    - dataframe (pd.DataFrame): DataFrame with added 'path' column
    containing image paths.
    """
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


def get_label_counts_and_print(dataframe: pd.DataFrame,
                               label_column: str
                               ) -> Dict[str, int]:
    """
    Get the counts of each label in the dataframe and print the results.

    Args:
    - dataframe (pd.DataFrame): DataFrame containing the dataset.
    - label_column (str): Name of the column containing labels in the dataset.

    Returns:
    - label_counts (Dict[str, int]): Dictionary containing label counts.
    """
    total_images = len(dataframe)  # Total number of images in the dataframe
    label_counts = dataframe[label_column].value_counts().to_dict()

    print(f"Total number of images: {total_images}")
    print(f"Number of unique labels: {len(label_counts)}")
    for label, count in label_counts.items():
        print(f"Label '{label}' has {count} images.")

    return label_counts


def calculate_image_statistics(dataframe: pd.DataFrame,
                               path_column: str = 'path'
                               ) -> Tuple[
                                int,
                                int,
                                Dict[str, List[Tuple[float, float]]]]:
    """
    Calculate statistics for the images in the DataFrame.

    Args:
    - dataframe (pd.DataFrame): DataFrame containing image paths.
    - path_column (str, optional): Name of the column containing
    image paths (default: 'path').

    Returns:
    - smallest_pixel (int): Smallest pixel value found in the images.
    - largest_pixel (int): Largest pixel value found in the images.
    - channel_values (Dict[str, List[Tuple[float, float]]]): Channel-wise
    mean and standard deviation values.
    """
    smallest_pixel = float('inf')
    largest_pixel = 0
    total_images = 0
    channel_values = {'R': [], 'G': [], 'B': []}
    # Iterate through each image in the dataframe
    for _, row in dataframe.iterrows():
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
                total_images += 1
                continue

            if len(image.shape) == 3 and image.shape[2] == 4:  # RGBA image
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

                if i == 0:
                    channel_label = 'R'
                elif i == 1:
                    channel_label = 'G'
                else:
                    channel_label = 'B'

                channel_values[channel_label].append((mean_val, std_val))

            total_images += 1

    return smallest_pixel, largest_pixel, channel_values


def print_image_statistics(smallest_pixel: int,
                           largest_pixel: int,
                           total_images: int,
                           channel_values: Dict
                           )-> None:
    """
    Print the image statistics.

    Args:
    - smallest_pixel (int): Smallest pixel value found in the images.
    - largest_pixel (int): Largest pixel value found in the images.
    - total_images (int): Total number of images processed.
    - channel_values (Dict[str, List[Tuple[float, float]]]): Channel-wise
    mean and standard deviation values.
    """
    print(f"Smallest pixel value: {smallest_pixel}")
    print(f"Largest pixel value: {largest_pixel}")
    print(f"Total images processed: {total_images}")

    if not channel_values:
        print("No images were processed.")
    else:
        print("Channel Statistics:")
        for channel, values in channel_values.items():
            print(f"Channel '{channel}':")
            for i, (mean, std) in enumerate(values):
                print(f"  Image {i+1} - Mean: {mean},"
                      f" Standard Deviation: {std}")


def format_image_statistics(smallest_pixel: int,
                            largest_pixel: int,
                            total_images: int,
                            channel_values: Dict
                            ) -> Dict:
    """
    Format the image statistics into a dictionary.

    Args:
    - smallest_pixel (int): Smallest pixel value found in the images.
    - largest_pixel (int): Largest pixel value found in the images.
    - total_images (int): Total number of images processed.
    - channel_values (Dict): Channel-wise mean and standard deviation values.

    Returns:
    - image_stats (Dict): Dictionary containing image statistics.
    """
    is_gray = len(channel_values) == 1 and 'R' in channel_values
    if is_gray:
        return {
            'smallest_pixel_value': smallest_pixel,
            'largest_pixel_value': largest_pixel,
            'total_images': total_images,
            'channel_statistics': {},
            'channels': 1
        }

    # Calculate average and standard deviation per channel
    channel_stats = {}
    for channel, values in channel_values.items():
        avg = np.mean([val[0] for val in values])
        std_dev = np.mean([val[1] for val in values])
        channel_stats[channel] = {'average': avg, 'std_dev': std_dev}

    return {
        'smallest_pixel_value': smallest_pixel,
        'largest_pixel_value': largest_pixel,
        'total_images': total_images,
        'channel_statistics': channel_stats,
        'channels': 3
    }


def image_analysis(dataframe: pd.DataFrame,
                   path_column: str = 'path'
                   ) -> Dict:
    """
    Analyze the images in the DataFrame and extract image statistics.

    Args:
    - dataframe (pd.DataFrame): DataFrame containing image paths.
    - path_column (str, optional): Name of the column containing
    image paths (default: 'path').

    Returns:
    - image_stats (Dict): Dictionary containing image statistics.
    """
    smallest_pixel, largest_pixel, channel_values = (
        calculate_image_statistics(dataframe, path_column))
    print_image_statistics(smallest_pixel,
                           largest_pixel,
                           len(dataframe),
                           channel_values)
    image_stats = format_image_statistics(smallest_pixel,
                                          largest_pixel,
                                          len(dataframe),
                                          channel_values)
    return image_stats


def check_images_existence(dataframe: pd.DataFrame,
                           path_column: str = 'path'
                           ) -> pd.DataFrame:
    """
    Check the existence of image files in the specified folder path.

    Args:
    - dataframe (pd.DataFrame): DataFrame containing image paths.
    - path_column (str, optional): Name of the column containing
    image paths (default: 'path').

    Returns:
    - cleaned_dataframe (pd.DataFrame): DataFrame with removed
    rows containing non-existent image paths.
    """
    cleaned_dataframe = dataframe.copy()
    removed_images = []

    # Iterate through each row in the DataFrame
    for index, row in dataframe.iterrows():
        image_path = row[path_column]

        if image_path is None or not os.path.exists(image_path):
            removed_images.append(row)
            cleaned_dataframe.drop(index, inplace=True)
            print(f"Image not found in folder: {image_path}")

    # Print removed lines
    if removed_images:
        print("Removed lines:")
        for row in removed_images:
            print(row)

    return cleaned_dataframe


def analyze_image_shapes(dataframe: pd.DataFrame,
                         min_shape: Tuple[int, int],
                         path_column: str = 'path'
                         ) -> Dict[str, int]:
    """
    Analyze the shapes of images in the DataFrame.

    Args:
    - dataframe (pd.DataFrame): DataFrame containing image paths.
    - min_shape (Tuple[int, int]): Minimum shape required for images.
    - path_column (str, optional): Name of the column containing
    image paths (default: 'path').

    Returns:
    - image_stats (Dict[str, int]): Dictionary containing image
    shape statistics.
    """
    total_images = 0
    total_height = 0
    total_width = 0
    smaller_than_x_count = 0

    # Iterate through each row in the dataframe
    for _, row in dataframe.iterrows():
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
    print(f"Average image shape - Height: "
          f"{average_height}, Width: {average_width}")

    print("Number of images with shape smaller "
          f"than {min_shape}: {smaller_than_x_count}")

    # Return results as a dictionary
    return {
        'average_height': average_height,
        'average_width': average_width,
        'smaller_than_x_count': smaller_than_x_count
    }
