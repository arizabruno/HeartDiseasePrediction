import pandas as pd
from utils import encode_categorical_features, remove_iqr_outliers

def load_data(file_path):
    """
    Loads data from a CSV file into a pandas DataFrame with specified column names.

    Parameters:
    file_path (str): The file path of the CSV file to be loaded.

    Returns:
    pandas.DataFrame: A DataFrame containing the loaded data.
    """
    column_names = [
        "age", "sex", "chest_pain_type", "resting_bps", "cholesterol",
        "fasting_blood_sugar", "resting_ecg", "max_heart_rate",
        "exercise_angina", "old_peak", "st_slope", "target"
    ]
    return pd.read_csv(file_path, names=column_names)


def preprocess_data(df):
    """
    Preprocesses the given DataFrame by encoding categorical columns and handling outliers.

    The function first converts specified columns to categorical data types.
    Then, it applies outlier removal and encoding of categorical features.

    Parameters:
    df (pandas.DataFrame): The DataFrame to preprocess.

    Returns:
    pandas.DataFrame: The preprocessed DataFrame.
    """
    categorical_cols = ['sex', 'chest_pain_type', 'fasting_blood_sugar', 'resting_ecg', 
                        'exercise_angina', 'st_slope', 'target']
    df[categorical_cols] = df[categorical_cols].astype('category')

    # Assuming remove_iqr_outliers and encode_categorical_features are defined elsewhere
    df = remove_iqr_outliers(df)  
    df = encode_categorical_features(df, "target")

    return df


if __name__ == "__main__":
    # Load data
    df = load_data('../data/raw/data.csv')

    # Preprocess data
    processed_df = preprocess_data(df)

    # Save the processed data
    processed_df.to_csv('../data/processed/data.csv', index=False)
