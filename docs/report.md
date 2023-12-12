# Heart Disease Prediction Project

## 1. Introduction

### Project Overview
This project aims to develop a predictive model to identify the likelihood of heart disease in patients. Leveraging machine learning techniques, the model will analyze various medical parameters to predict heart disease risk. This tool's primary goal is to aid healthcare professionals in early diagnosis and personalized patient care.

### Background
Heart disease, encompassing a range of cardiovascular conditions, is a leading cause of mortality worldwide. Early detection and preventive measures can significantly improve patient outcomes. Data science plays a crucial role in healthcare by enabling the analysis of complex medical data to identify patterns and predict health risks.

### Project Goals
The main objectives of this project are:
- To predict the likelihood of heart disease in patients using machine learning algorithms.
- To provide a tool for healthcare professionals that can assist in early intervention and better patient management.
- To analyze various medical parameters and their correlation with heart disease.

## 2. Data Source and Description

### Data Acquisition
The dataset for this project was sourced from [OpenML](https://www.openml.org/search?type=data&status=active&id=43672), specifically designed for heart disease prediction. It encompasses a comprehensive set of medical parameters collected from various patients.

### Dataset Description
The dataset comprises several features that are critical in diagnosing heart disease, including:
- **Age**: Patient's age in years.
- **Sex**: Patient's sex (1 = male, 0 = female).
- **Chest Pain Type**: Type of chest pain experienced by the patient.
- **Resting Blood Pressure**: Resting blood pressure value of the patient.
- **Cholesterol**: Serum cholesterol level.
- **Fasting Blood Sugar**: Indicates if fasting blood sugar is higher than 120 mg/dl (1 = true, 0 = false).
- **Resting ECG**: Results of the electrocardiogram on rest.
- **Max Heart Rate**: Maximum heart rate achieved.
- **Exercise Angina**: Angina induced by exercise (1 = yes, 0 = no).
- **Oldpeak**: ST depression induced by exercise relative to rest.
- **ST Slope**: The slope of the peak exercise ST segment.
- **Target**: Presence of heart disease (1 = present, 0 = absent).

The dataset consists of [number of samples] instances with [number of features] features, providing a robust foundation for building a predictive model.


## Handling Outliers

The following process has been implemented to identify and handle outliers in numerical columns of the dataset. The approach is based on the Interquartile Range (IQR) method, a robust statistical technique used to detect outliers.

**Step 1: Identify Numerical Columns**

First, we extract a list of all numerical columns in the dataset since our focus is on numerical outliers.


**Step 2: Calculate IQR for Each Column**

For each numerical column, we calculate the Interquartile Range (IQR), which is the difference between the 75th percentile (Q3) and the 25th percentile (Q1).

**Step 3: Define Outlier Limits**

Using the IQR, we define the limits beyond which data points will be considered outliers. Data points below the lower limit or above the upper limit are candidates for outlier treatment.

- **Lower Limit:** Q1 - 1.5 * IQR
- **Upper Limit:** Q3 + 1.5 * IQR

**Step 4: Apply Capping**

We cap the outliers by applying the lower and upper limits. Values beyond these limits are set to the limits themselves, thus minimizing the impact of extreme outliers.

Upper Capping: Values greater than the upper limit are set to the upper limit.
Lower Capping: Values lower than the lower limit are set to the lower limit.

**Step 5: Update the Dataset**

The original dataset is updated with the capped values, ensuring that the outliers do not disproportionately affect the subsequent analysis.