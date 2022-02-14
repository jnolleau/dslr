import numpy as np


def get_numeric_col_names(df) -> list:
    return df.select_dtypes(include=np.number).columns.tolist()


def standardize_feature(feature):
    return (feature - feature.mean())/feature.std()


def standardize_df(df):
    num_features = get_numeric_col_names(df)
    for feature in num_features:
        df[feature] = standardize_feature(df[feature])
    return df
