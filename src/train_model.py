import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import joblib

def load_preprocessed_data(file_path):
    """
    Loads preprocessed data from a specified file path.

    Parameters:
    file_path (str): The file path of the preprocessed data.

    Returns:
    pandas.DataFrame: A DataFrame containing the preprocessed data.
    """
    return pd.read_csv(file_path)


def split_data(df, target_column):
    """
    Splits the DataFrame into training and testing sets.

    Parameters:
    df (pandas.DataFrame): The DataFrame to split.
    target_column (str): The name of the target column.

    Returns:
    X_train, X_test, y_train, y_test: Split datasets.
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    # Split the data into training and testing sets (80% train, 20% test)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def scale_features(X_train, X_test, numerical_cols):
    """
    Applies feature scaling to the numerical features of the training and test sets.

    Parameters:
    X_train (pandas.DataFrame): Training features.
    X_test (pandas.DataFrame): Test features.
    numerical_cols (list): List of names of the numerical columns to scale.

    Returns:
    Scaled versions of X_train and X_test, with one-hot encoded features unchanged.
    """
    # Create a column transformer which only scales the numerical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols)
        ],
        remainder='passthrough'  # Leave one-hot encoded columns unchanged
    )

    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    return X_train_scaled, X_test_scaled

def train_model(X_train, y_train):
    """
    Trains the machine learning model on the training data.

    Parameters:
    X_train: Training features.
    y_train: Training target.

    Returns:
    The trained model.
    """
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model
    
    
if __name__ == "__main__":
    # Load preprocessed data
    df = load_preprocessed_data('../data/processed/data.csv')
    X_train, X_test, y_train, y_test = split_data(df, 'target')

    # Identify numerical columns (exclude one-hot encoded columns)
    numerical_cols = ["age","resting_bps","cholesterol","max_heart_rate","old_peak",]

    # Apply feature scaling to numerical features only
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test, numerical_cols)
    
    # Save the scaled training and test sets
    pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv('../data/train/X_train_scaled.csv', index=False)
    pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv('../data/test/X_test_scaled.csv', index=False)
    y_train.to_csv('../data/train/y_train.csv', index=False)
    y_test.to_csv('../data/test/y_test.csv', index=False)

    # Train the model using the scaled data
    model = train_model(X_train_scaled, y_train)

    # Save the trained model
    joblib.dump(model, '../models/model_v1.joblib')
