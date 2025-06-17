# Substance Use and Mental Health Prediction

-----

## Project Overview

This project aims to predict various addiction-related and mental health outcomes using the National Survey on Drug Use and Health (NSDUH) dataset. We explore different machine learning classification models, including AdaBoost, Stacking Classifier, Neural Network (MLP), and Bagging Classifier, to identify individuals at risk and determine the most impactful features for these predictions.

## Table of Contents

  - [Project Overview](https://www.google.com/search?q=%23project-overview)
  - [Dataset](https://www.google.com/search?q=%23dataset)
  - [Installation](https://www.google.com/search?q=%23installation)
  - [Usage](https://www.google.com/search?q=%23usage)
  - [Data Preprocessing](https://www.google.com/search?q=%23data-preprocessing)
  - [Feature Engineering and Selection](https://www.google.com/search?q=%23feature-engineering-and-selection)
  - [Models and Evaluation](https://www.google.com/search?q=%23models-and-evaluation)
  - [Conclusion](https://www.google.com/search?q=%23conclusion)

-----

## Dataset

The dataset used in this project is `1999-2022nah.csv`. It contains information related to substance use and mental health from the National Survey on Drug Use and Health (NSDUH) spanning from 1999 to 2022.

The dataset includes columns such as:

  - `outname`: Specific outcome name (e.g., 'alcohol use in the past month'). This is the primary target for categorization and prediction.
  - `pyearnm`: Period name (e.g., '1999-2000').
  - `area`: Geographic area.
  - `outcome`: Outcome code.
  - `stname`: State name.
  - `pyear`: Survey year.
  - `agegrp`: Age group.
  - `est_total`, `low_total`, `up_total`: Estimated total, lower and upper bounds.
  - `BSAE`, `low_sae`, `up_sae`: Bayesian Small Area Estimation and its bounds.
  - `ste_total`, `ste_sae`: Standard error of total and SAE.
  - `state`: State code.
  - `gen_corr`: General correlation.

-----

## Installation

To run this project, you'll need Python and several libraries. You can install the required packages using pip:

```bash
pip install pandas scikit-learn matplotlib seaborn
```

## Usage

1.  **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Place the dataset:**

    Ensure the `1999-2022nah.csv` file is in the root directory of the project.

3.  **Run the script:**

    You can execute the Python script containing the code.

The script will perform data loading, preprocessing, feature engineering, model training, and evaluation, displaying various plots and classification reports.

-----

## Data Preprocessing

The preprocessing steps involve:

  - **Loading Data**: The `1999-2022nah.csv` file is loaded into a Pandas DataFrame.
  - **Categorization of `outname`**: The original `outname` column, which contains granular descriptions of various outcomes, is mapped into broader, more manageable categories: `suicide_or_self_harm`, `mental_disorder`, `substance_use`, and `risk_perceptions`. This helps in creating a more general classification task.
  - **Filtering `agegrp`**: The dataset is filtered to include only specific age groups (1.0, 0.0, 5.0) to focus the analysis.
  - **Imputation**: Missing values in `ste_total`, `ste_sae`, and `gen_corr` columns are imputed using the mean strategy.
  - **Handling Categorical Features**: Categorical features like `outcome` are converted into numerical representations using label encoding or one-hot encoding for model compatibility.

-----

## Feature Engineering and Selection

  - **Feature Importance**: A **Random Forest Classifier** is used to determine the importance of various features in predicting the `outname` categories.
  - **Feature Selection**: Based on feature importances, a subset of the most impactful features (e.g., `BSAE`, `low_sae`, `up_sae`, `outcome`) is selected for model training to reduce dimensionality and potentially improve performance.

-----

## Models and Evaluation

Several classification models are implemented and evaluated:

### 1\. AdaBoost Classifier

  - **Description**: An ensemble meta-estimator that fits a sequence of weak learners (e.g., decision trees) on re-weighted versions of the data.
  - **Evaluation**: Performance is assessed using a classification report (precision, recall, f1-score) and a confusion matrix.

### 2\. Stacking Classifier

  - **Description**: Combines the predictions of multiple base estimators (e.g., Decision Tree, Logistic Regression) and uses a final estimator (meta-learner) to make the final prediction.
  - **Evaluation**: Classification report and confusion matrix are used to evaluate its performance.

### 3\. Neural Network (MLPClassifier)

  - **Description**: A multi-layer perceptron neural network with specified hidden layers and maximum iterations.
  - **Evaluation**: Comprehensive evaluation including classification report, confusion matrix heatmap, and distribution of predictions.

### 4\. Bagging Classifier

  - **Description**: An ensemble meta-estimator that fits base classifiers on random subsets of the original dataset, then aggregates their individual predictions (e.g., by voting or averaging) to form a final prediction.
  - **Evaluation**: Classification report and confusion matrix.

For each model, **classification reports** provide detailed metrics (precision, recall, f1-score, support) for each class, and **confusion matrices** visually represent the model's performance in terms of true positives, true negatives, false positives, and false negatives.

-----

## Conclusion

This project demonstrates a robust approach to classifying addiction-related and mental health outcomes using the NSDUH dataset. The process involves comprehensive data preprocessing, intelligent feature selection, and the application of various machine learning models.

The **Stacking Classifier** and **Neural Network (MLP)** models exhibited exceptionally high performance (near 1.00 for precision, recall, and f1-score across all classes) in the final evaluations on the selected features. This indicates their strong ability to differentiate between the defined outcome categories given the chosen features. The **Bagging Classifier** also showed robust performance, achieving 1.00 for all metrics, suggesting it's a very stable and accurate model for this dataset. The **AdaBoost Classifier**, while performing well overall, had some limitations with certain minority classes, as indicated by lower precision and recall for classes 0 and 1.

The `BSAE`, `low_sae`, `up_sae`, and the encoded `outcome` column were identified as the most important features, highlighting their significance in predicting the categorized `outname`.

This analysis provides valuable insights into the factors contributing to various public health outcomes and showcases the effectiveness of advanced machine learning techniques in this domain.

Further work could involve:

  - Exploring other feature engineering techniques.
  - Hyperparameter tuning for all models to potentially improve `AdaBoost`'s performance.
  - Investigating the impact of different data imputation strategies.
  - Deploying the best-performing model as an API for real-time predictions.
