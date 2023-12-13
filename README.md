# Heart Disease Prediction 

<p align="center">
    <img src="img/logo.png" alt="Logo" width="200"/>
</p>

## Introduction
This project focuses on developing a machine learning model to predict heart disease using various medical parameters. The goal is to aid healthcare professionals in early diagnosis and treatment.

## Project Overview
- **Data Source**: The dataset, sourced from [OpenML](https://www.openml.org/search?type=data&status=active&id=43672), includes features like age, sex, cholesterol levels, etc.
- **Techniques**: The project uses data preprocessing (handling outliers, feature scaling, and encoding), exploratory data analysis, and model development.
- **Model**: A RandomForestClassifier was trained and evaluated.

## Data Preprocessing
- **Outlier Handling**: Applied IQR-based capping.
- **Feature Scaling**: StandardScaler for numerical features.
- **Encoding**: OneHotEncoder for categorical features.

## Model Evaluation Explained

The performance of our heart disease prediction model is measured using several metrics:

### Accuracy (95.38%)
- **What it Means**: Accuracy tells us the percentage of total predictions our model got right. 
- **In Context**: In our case, 95.38% accuracy means that the model correctly identified heart disease (or the lack of it) in 95.38% of the cases.

### Precision (94.78%)
- **What it Means**: Precision indicates the proportion of positive identifications that were actually correct.
- **Example**: If our model identifies 100 patients as having heart disease, about 94.78 of them truly have the disease.

### Recall (96.95%)
- **What it Means**: Recall measures the proportion of actual positives that were identified correctly.
- **Example**: If there are 100 patients who actually have heart disease, the model correctly identifies about 96.95 of them.

### F1 Score (95.85%)
- **What it Means**: The F1 Score is the harmonic mean of Precision and Recall, providing a balance between the two.
- **In Context**: A high F1 Score, like ours at 95.85%, suggests a well-balanced model in terms of both precision and recall.

Overall, these metrics indicate a high level of reliability in our model's ability to predict heart disease, making it a valuable tool in medical diagnostics.

### Note on Evaluation Results Discrepancy
The slight discrepancy in model evaluation metrics between the Jupyter notebook and the refactored scripts can be attributed to factors such as random data splits, model initialization, or minor differences in data preprocessing. 


## Project Structure and Organization

The Heart Disease Prediction Project is meticulously organized to ensure clarity and ease of navigation:

- **Data Directory (`/data`)**: Contains the raw dataset (`/raw`), processed data (`/processed`), and split training and test sets (`/train` and `/test`).
- **Notebooks (`/notebooks`)**: Jupyter notebooks for exploratory data analysis and detailed project walkthrough (`analysis.ipynb`).
- **Source Code (`/src`)**: Python scripts for data preprocessing (`data_preprocessing.py`), model training (`train_model.py`), and evaluation (`evaluate_model.py`), along with utility functions (`utils.py`).
- **Models (`/models`)**: Houses the trained model file (`model_v1.joblib`), ready for predictions.
- **Notes (`/notes`)**: Miscellaneous notes related to the project.

This structure aids in the systematic progression of the project, from data handling to final model evaluation, ensuring a smooth workflow for any contributors or reviewers.


## How to Run the Project


To run this project and explore the Heart Disease Prediction analysis, follow these steps:

1. **Clone the Repository**: Clone this repository to your local machine using `git clone [repository URL]`.

2. **Install Dependencies**: Navigate to the project directory and install the required dependencies using: `pip install -r requirements.txt`
1. **Explore the Jupyter Notebook**:
- Open the Jupyter notebook (`notebooks/analysis.ipynb`) in a Jupyter environment.
- You can use Jupyter Lab, Jupyter Notebook, or any IDE that supports Jupyter notebooks.
- Run each cell in the notebook to see the analysis process, from data loading and preprocessing to model training and evaluation.
1. **Run Python Scripts (Optional)**:
- To execute individual scripts, navigate to the `src/` directory.
- Run the Python scripts for specific tasks, like `python data_preprocessing.py` for preprocessing, `python train_model.py` for training the model, and `python evaluate_model.py` for model evaluation.

Following these steps, you can replicate the analysis, explore the results, and even extend the project with your insights.


## Further Work
Future enhancements include deeper data exploration, advanced model tuning, and deployment for practical use.

## License
This project is licensed under MIT License.

## Author
* Bruno Ariza


