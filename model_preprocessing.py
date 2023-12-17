def categorial2number(dataframe, labels_dict, label_column):
    label_dataframe = [labels_dict[item] if item in labels_dict else item for item in list(dataframe[label_column])]
    return label_dataframe


def dflabel2number(df_lists, labels_dict, label_column):
    df1 = categorial2number(df_lists[0], labels_dict, label_column)
    df2 = categorial2number(df_lists[1], labels_dict, label_column)
    df3 = categorial2number(df_lists[2], labels_dict, label_column)

    return df1, df2, df3


def labels2dict(dataframe, label_column):
    labels = (dataframe[label_column].unique())
    labels_dict = {}
    for index, label in enumerate(labels):
        labels_dict[label] = index
        
    return labels_dict



def calculate_weights(train_df, label_dict, dict_train_qntd):
    num_classes = len(label_dict)
    total_samples = len(train_df)

    weights = []
    for label, idx in label_dict.items():
        label_count = dict_train_qntd.get(label, 0)
        weight = total_samples / (label_count * num_classes)
        weights.append(weight)

    return weights
