import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

def get_mising_data(df: pd.DataFrame):
    total_missing = df.isnull().sum().sort_values(ascending=False)
    whole_data_count = df.isnull().count()
    missing_persent = (total_missing / whole_data_count).sort_values(ascending=False)
    total_missing = total_missing[total_missing > 0]
    missing_persent = missing_persent[missing_persent > 0]
    missing_data = pd.concat([total_missing, missing_persent], axis=1, keys=['Total', 'Percent'])
    return missing_data

def delete_nans(train_df: pd.DataFrame, tresh: float):
    new_df = train_df.copy()
    train_miss = get_mising_data(new_df)

    deleted_train_labels = train_miss[train_miss['Percent'] > tresh].index
    new_train = new_df.drop(labels=deleted_train_labels, axis=1)
    
    return new_train

def fill_nans(train_df: pd.DataFrame):
    new_train = train_df.copy()
    new_train = new_train.fillna('None')
    return new_train


def convert_to_numeric(train_df: pd.DataFrame):
    new_train = train_df.copy()
    LE = LabelEncoder()
    for feature in new_train.columns:
        if (new_train[feature].dtype == 'object'):
            new_train[feature] = LE.fit_transform(new_train[feature])
    return new_train
    