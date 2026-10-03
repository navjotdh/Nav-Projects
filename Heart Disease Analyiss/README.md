# Heart Disease Analysis Using Python

## Project Overview

This project explores patterns associated with heart disease using the Cleveland Heart Disease dataset. I used Python to perform exploratory data analysis, visualize relationships between patient characteristics and heart disease, and build a logistic regression model to classify patients based on the available clinical and demographic variables.

The project focuses on making the analysis interpretable by first examining patterns in the data and then evaluating how well a basic machine-learning model can classify heart disease cases.

## Dataset

The dataset contains 297 patient records and 14 columns, including the heart disease outcome.

The analysis uses 13 predictor variables:

- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol
- Fasting blood sugar
- Resting ECG results
- Maximum heart rate
- Exercise-induced angina
- ST depression (oldpeak)
- ST slope
- Number of major vessels
- Thalassemia

The target variable is `condition`:

- `0` = No heart disease
- `1` = Heart disease

The dataset was checked for missing values and duplicate records. No missing values or duplicate rows were identified.

## Exploratory Data Analysis

The dataset contained 160 patients without heart disease and 137 patients with heart disease.

Several differences appeared during exploratory analysis.

### Heart Disease by Sex

The dataset contained 201 male and 96 female patients.

- 55.72% of male patients had heart disease.
- 26.04% of female patients had heart disease.


### Age

Patients with heart disease tended to be older within both sex groups.

Among female patients, the average age was 59.08 years for those with heart disease compared with 54.58 years for those without heart disease.

Among male patients, the average age was 56.24 years for those with heart disease compared with 51.10 years for those without heart disease.


### Chest Pain Type

Chest pain type showed a noticeable relationship with heart disease status. In the dataset, the asymptomatic chest pain category contained substantially more heart disease cases than the other chest pain categories.

### Maximum Heart Rate

Average maximum heart rate was:

- 139.11 bpm for patients with heart disease
- 158.58 bpm for patients without heart disease

### Resting Blood Pressure

Average resting blood pressure was:

- 134.64 mmHg for patients with heart disease
- 129.18 mmHg for patients without heart disease

### Cholesterol

Average cholesterol was:

- 251.85 mg/dL for patients with heart disease
- 243.49 mg/dL for patients without heart disease

## Logistic Regression Model

After exploratory analysis, I built a logistic regression model using the 13 predictor variables.

The dataset was divided into:

- 70% training data
- 30% testing data

I used a stratified train-test split so that the proportion of heart disease cases remained similar between the training and testing datasets.

### Model Performance

The logistic regression model achieved an accuracy of 87.78% on the test set.

Confusion matrix:

|                   | Predicted No Disease | Predicted Disease |
| Actual No Disease | 46                   | 2                 |
| Actual Disease    | 9                    | 33                |

For patients with heart disease, the model achieved:

- Precision: 94%
- Recall: 79%
- F1-score: 86%

Although overall accuracy was approximately 88%, the model failed to identify 9 of the 42 heart disease cases in the test set.

## Tools and Libraries

- Python
- pandas
- Matplotlib
- scikit-learn



This project helped me practice the complete workflow of a small data-analysis and machine-learning project, including data inspection, data-quality checks, exploratory analysis, visualization, train-test splitting, logistic regression, and model evaluation.

It also reinforced the importance of looking beyond overall accuracy when evaluating a classification model. In this dataset, examining precision, recall, F1-score, and the confusion matrix provided a clearer picture of where the model performed well and where it made errors.

## Limitations

This analysis uses a relatively small dataset of 297 observations, so the model results should not be interpreted as evidence of clinical performance.

The project is intended as an exploratory data-analysis and machine-learning exercise rather than a medical diagnostic tool.

## Data Source

Cleveland Heart Disease dataset available through Kaggle: Heart Disease Cleveland UCI.
