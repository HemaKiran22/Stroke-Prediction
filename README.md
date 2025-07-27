# Stroke Risk Prediction using Machine Learning
This project develops a machine learning model to predict the likelihood of a patient having a stroke based on a range of clinical, demographic, and lifestyle factors. The primary goal is to create a decision-support tool that can help identify at-risk individuals , enabling timely medical intervention and potentially reducing morbidity and mortality
## Project Goal
The main objective is to build a robust classification model for stroke risk. This project focuses not just on building a model, but on addressing the specific data challenges inherent in medical datasets, such as severe class imbalance and missing values.
## Methodology

The project follows a standard machine learning pipeline consisting of the following steps:

### 1. Data Cleaning

- Removed the `id` column as it is irrelevant for prediction.
- Removed ambiguous or inconsistent entries (e.g., a single 'Other' gender entry) to avoid modeling issues.

### 2. Preprocessing

A `ColumnTransformer` pipeline was built to handle mixed data types:

- **Numerical Features** (`age`, `avg_glucose_level`, `bmi`):
  - Imputed using `IterativeImputer`, a multivariate approach that models missing values based on other columns.
  - Scaled using `StandardScaler` to normalize the features.

- **Categorical Features**:
  - Imputed with the most frequent value using `SimpleImputer`.
  - Encoded using `OneHotEncoder`.

### 3. Handling Class Imbalance

- The dataset is highly imbalanced: only **4.9%** of patients had a stroke.
- Used **SMOTE (Synthetic Minority Over-sampling Technique)** to generate synthetic examples of the minority class.
- SMOTE was integrated into the pipeline to ensure the model learns from a balanced dataset during training.

### 4. Model Training

- Used a **Random Forest Classifier**, a tree-based ensemble model.
- Random Forest is well-suited for this task due to:
  - Its robustness to overfitting
  - Ability to handle both numerical and categorical data
  - Built-in feature importance capabilities

### 5. Evaluation

- Accuracy was **not** used due to class imbalance.
- Evaluation metrics focused on:
  - **Recall (Sensitivity)**: Minimize false negatives
  - **Precision**: Ensure predictions are correct
  - **F1-Score**: Balance of precision and recall
- A **confusion matrix** and **classification report** were also generated to evaluate performance across both classes.

