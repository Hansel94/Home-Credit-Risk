from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer


def preprocess_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre processes data for modeling. Receives train, val and test dataframes
    and returns numpy ndarrays of cleaned up dataframes with feature engineering
    already performed.

    Arguments:
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        test_df : pd.DataFrame

    Returns:
        train : np.ndarrary
        val : np.ndarrary
        test : np.ndarrary
    """
    # Print shape of input data
    print("Input train data shape: ", train_df.shape)
    print("Input val data shape: ", val_df.shape)
    print("Input test data shape: ", test_df.shape, "\n")

    # Make a copy of the dataframes
    working_train_df = train_df.copy()
    working_val_df = val_df.copy()
    working_test_df = test_df.copy()

    # 1. Correct outliers/anomalous values in numerical
    # columns (`DAYS_EMPLOYED` column).
    working_train_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_val_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_test_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)

    # 2. Encode string categorical features:
    #     - for features with 2 categories binary encoding is used
    #     - For features with more than 2 categories, one-hot encoding is used
    df_cats = working_train_df.select_dtypes(object)
    bin_cols = []
    mv_cols = []
    for column in df_cats.columns:
        if df_cats[column].value_counts().count() == 2:
            bin_cols.append(column)
        else:
            mv_cols.append(column)
            
    bin_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
    one_hot = OneHotEncoder(handle_unknown='ignore')
    transformer = ColumnTransformer([('bin_enc', bin_enc, bin_cols),
                                     ('one_hot', one_hot, mv_cols)],
                                     remainder='passthrough')
                                     
    working_train_df = transformer.fit_transform(working_train_df)
    working_val_df = transformer.transform(working_val_df)
    working_test_df = transformer.transform(working_test_df)


    # 3. Impute values for all columns with missing data using median as imputing value
    imputer = SimpleImputer(strategy='median')
    working_train_df = imputer.fit_transform(working_train_df)
    working_val_df = imputer.transform(working_val_df)
    working_test_df = imputer.transform(working_test_df)

    # 4. Feature scaling with Min-Max scaler
    scaler = MinMaxScaler()
    working_train_df = scaler.fit_transform(working_train_df)
    working_val_df = scaler.transform(working_val_df)
    working_test_df = scaler.transform(working_test_df)

    return working_train_df, working_val_df, working_test_df
