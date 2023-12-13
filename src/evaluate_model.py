import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib


def load_model(model_path):
    """
    Load a trained model from a specified file path.

    Parameters:
    model_path (str): The file path where the model is saved.

    Returns:
    Trained model.
    """
    return joblib.load(model_path)

def load_scaled_test_data(X_test_path, y_test_path):
    """
    Load scaled test data from specified file paths.

    Parameters:
    X_test_path (str): File path for scaled test features.
    y_test_path (str): File path for test target.

    Returns:
    X_test, y_test: Scaled test features and test target.
    """
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)
    return X_test, y_test


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the performance of the model on the test data.

    Parameters:
    model: The trained model.
    X_test (DataFrame): Test features.
    y_test (Series): Test target.
    """
    X_test_array = X_test.to_numpy()
    predictions = model.predict(X_test_array)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    
    accurancy_formatted = "{:.2%}".format(accuracy)
    precision_formatted = "{:.2%}".format(precision)
    recall_formatted = "{:.2%}".format(recall)
    f1_formatted = "{:.2%}".format(f1)

    print(f"Accuracy: {accurancy_formatted}")
    print(f"Precision: {precision_formatted}")
    print(f"Recall: {recall_formatted}")
    print(f"F1 Score: {f1_formatted}")


if __name__ == "__main__":
    # Load scaled test data and model
    X_test, y_test = load_scaled_test_data('../data/test/X_test_scaled.csv', '../data/test/y_test.csv')
    model = load_model('../models/model_v1.joblib')

    # Evaluate the model
    evaluate_model(model, X_test, y_test)