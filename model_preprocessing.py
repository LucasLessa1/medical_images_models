def categorial2number(dataframe, labels_dict, label_column):
    label_dataframe = [labels_dict[item] if item in labels_dict else item for item in list(dataframe[label_column])]
    return label_dataframe


def dflabel2number(df_lists, labels_dict, label_column):
    df1 = categorial2number(df_lists[0], labels_dict, label_column)
    df2 = categorial2number(df_lists[1], labels_dict, label_column)
    df3 = categorial2number(df_lists[2], labels_dict, label_column)

    return df1, df2, df3