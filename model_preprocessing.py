import pandas as pd
from typing import Dict, List, Tuple


def categorial2number(dataframe: pd.DataFrame,
                      labels_dict: Dict[str, int],
                      label_column: str
                      ) -> List[int]:
    """
    Convert categorical labels in a DataFrame column to numerical labels
    based on a dictionary mapping.

    Args:
    - dataframe (pd.DataFrame): The DataFrame containing the labels.
    - labels_dict (dict): A dictionary mapping categorical labels to
    numerical labels.
    - label_column (str): The name of the column containing the
    categorical labels.

    Returns:
    - label_dataframe (List[int]): A list of numerical labels corresponding
    to the input categorical labels.
    """
    label_dataframe = [labels_dict[item]
                       if item in labels_dict else item
                       for item in list(dataframe[label_column])]
    return label_dataframe


def dflabel2number(df_lists: List[pd.DataFrame],
                   labels_dict: Dict[str, int],
                   label_column: str
                   ) -> Tuple[List[int], List[int], List[int]]:
    """
    Convert categorical labels in multiple DataFrames to numerical
    labels based on a dictionary mapping.

    Args:
    - df_lists (List[pd.DataFrame]): A list of DataFrames containing
    the labels.
    - labels_dict (dict): A dictionary mapping categorical labels to
    numerical labels.
    - label_column (str): The name of the column containing the
    categorical labels.

    Returns:
    - Tuple of lists: A tuple containing lists of numerical labels
    corresponding to the input categorical labels
      for each DataFrame in df_lists.
    """
    df1 = categorial2number(df_lists[0], labels_dict, label_column)
    df2 = categorial2number(df_lists[1], labels_dict, label_column)
    df3 = categorial2number(df_lists[2], labels_dict, label_column)

    return df1, df2, df3


def labels2dict(dataframe: pd.DataFrame, label_column: str) -> Dict[str, int]:
    """
    Create a dictionary mapping unique categorical labels to numerical labels.

    Args:
    - dataframe (pd.DataFrame): The DataFrame containing the labels.
    - label_column (str): The name of the column containing the
    categorical labels.

    Returns:
    - labels_dict (dict): A dictionary mapping unique categorical labels
    to numerical labels.
    """
    labels = (dataframe[label_column].unique())
    labels_dict = {}
    for index, label in enumerate(labels):
        labels_dict[label] = index

    return labels_dict


def calculate_weights(train_df: pd.DataFrame,
                      label_dict: Dict[str, int],
                      dict_train_qntd: Dict[str, int]
                      ) -> List[float]:
    """
    Calculate class weights for imbalanced classification.

    Args:
    - train_df (pd.DataFrame): The training DataFrame containing
    the labels.
    - label_dict (dict): A dictionary mapping categorical labels
    to numerical labels.
    - dict_train_qntd (dict): A dictionary mapping categorical
    labels to the number of samples for each label.

    Returns:
    - weights (List[float]): A list of class weights for each label.
    """
    num_classes = len(label_dict)
    total_samples = len(train_df)

    weights = []
    for label, _ in label_dict.items():
        label_count = dict_train_qntd.get(label, 0)
        weight = total_samples / (label_count * num_classes)
        weights.append(weight)

    return weights
