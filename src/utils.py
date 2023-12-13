import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def remove_iqr_outliers(df):
    """
    Removes outliers from the numerical columns in a pandas DataFrame based on the Interquartile Range (IQR) method.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data
    
    Returns:
    pandas.DataFrame: A DataFrame with outliers removed based on the IQR method.
    """
    
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    numerical_cols = numerical_columns.to_list()
    
    for column in numerical_cols:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        return filtered_df

def encode_categorical_features(df, target):
    """
    Encodes categorical features in a DataFrame using one-hot encoding.

    This function automatically identifies categorical columns (excluding the target variable),
    applies one-hot encoding, and returns a DataFrame with these encoded features
    replacing the original categorical features.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    target (str): The name of the target column to exclude from encoding.

    Returns:
    pandas.DataFrame: A DataFrame with categorical variables one-hot encoded.
    """

    # Identify categorical columns excluding the target variable
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.to_list()
    categorical_columns.remove(target)
    
    # One-hot encode the categorical data
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_cols = encoder.fit_transform(df[categorical_columns])
    encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_columns))

    # Drop original categorical columns and concatenate encoded columns
    df = df.drop(categorical_columns, axis=1)
    df = pd.concat([df, encoded_df], axis=1)

    return df