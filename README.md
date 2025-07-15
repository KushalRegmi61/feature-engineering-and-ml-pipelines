# Week 8: Feature Engineering & Model Selection

**Fusemachines AI Fellowship**

This week focused on **building a complete machine learning workflow** using the Titanic dataset—from preprocessing and feature engineering to model tuning and evaluation. The main goal was to extract meaningful features from structured data and compare the performance of different models through systematic hyperparameter optimization.

---

## Repository Structure

```

feature-selection-and-model-selection/
├── titanic_dataset_modeling.ipynb
├── LICENSE
└── README.md

```

---

## Key Concepts Covered

* **Missing Value Imputation**: Replacing missing values using statistical methods like median and mode.
* **Feature Engineering**:
  - Title extraction from names using regex
  - Grouping rare titles under a common category (`Special`)
  - Binning `Age` and `Fare` into quartile-based categories
  - Creating new features like `Fam_Size` and binary indicators (`Has_Cabin`)
* **One-Hot Encoding**: Converting categorical features into numerical ones while avoiding multicollinearity.
* **Model Training and Evaluation**:
  - Decision Tree Classifier with `RandomizedSearchCV`
  - Random Forest Classifier with parameter tuning
  - XGBoost Classifier using `BayesSearchCV` for Bayesian optimization
* **Cross-Validation**: 5-fold CV to ensure robust evaluation
* **Model Selection**: Comparing models based on best CV score and optimal hyperparameters

---

## File Overview

### [`titanic_modeling.ipynb`](/notebooks/titanic_modeling.ipynb)

* **Objective**: Build a predictive model for Titanic passenger survival by applying feature engineering and optimizing model hyperparameters.

* **What it does**:

  * Performs EDA and handles missing values in `Age`, `Fare`, and `Embarked`.
  * Extracts and normalizes titles from passenger names and groups rare titles as `Special`.
  * Creates new engineered features:
    - `CatAge` and `CatFare` (binned categories)
    - `Fam_Size` (sum of `SibSp` and `Parch`)
    - `Has_Cabin` (binary indicator for missing cabin data)
  * Applies one-hot encoding on all categorical variables.
  * Trains a **Decision Tree Classifier** using `RandomizedSearchCV`.
    - Best `max_depth`: 6
    - Best CV Score: 0.817
  * Trains a **Random Forest Classifier** with hyperparameter tuning.
    - Best Params: `n_estimators=100`, `max_features='log2'`, `max_depth=5`
    - Best CV Score: 0.834
  * Trains an **XGBoost Classifier** using Bayesian Optimization (`BayesSearchCV`).
    - Tuned over `max_depth`, `n_estimators`, `learning_rate`
    - Used log-uniform distribution for learning rate and ran 20 iterations

* **Key takeaway**: Proper feature engineering combined with targeted hyperparameter tuning leads to more accurate and generalizable models, even with traditional ML algorithms.

---

## Summary

This week emphasized the **importance of preprocessing and engineered features** in classical machine learning tasks. By applying transformations such as binning, title grouping, and encoding, we greatly enhanced model performance. Using tools like `RandomizedSearchCV` and `BayesSearchCV`, we systematically explored the hyperparameter space and selected the most optimal models based on cross-validation performance.

---

## Links

* [Main Fellowship Repository](https://github.com/KushalRegmi61/AI_Fellowship_FuseMachines)


