# Experiment 1 — Working with Python Packages: NumPy, SciPy, Scikit-Learn, Matplotlib

## Overview

This experiment serves as the foundational laboratory session for **ICS1512 – Machine Learning Algorithms Laboratory**. It explores the scientific computing, data manipulation, machine learning, and visualization ecosystem in Python. The objective is to implement reusable Exploratory Data Analysis (EDA) pipelines for both tabular and image datasets, analyze statistical distributions, and systematically map real-world datasets to appropriate machine learning tasks, algorithms, and feature-selection techniques.

## Objectives

- Explore core data structures and methods across **NumPy**, **Pandas**, **SciPy**, **Scikit-learn**, **Matplotlib**, and **Seaborn**.
- Design reusable, modular EDA routines for tabular and image domains.
- Profile real-world public datasets covering diverse data modalities (tabular clinical records, financial applications, and raster image data).
- Diagnose data quality issues, such as missing values, zero-value anomalies, skewness, and class imbalance.
- Map datasets to their respective machine learning paradigm (Supervised Classification/Regression, Unsupervised) and identify suitable feature-selection methods.

## Concepts Covered

- Multi-dimensional array manipulation and vectorized computations (NumPy).
- DataFrame operations, type inspection, missing-value profiling, and summary statistics (Pandas).
- Distribution skewness and summary statistics (SciPy / Pandas).
- Data visualization: feature histograms, correlation heatmaps, class distribution bar plots, pixel intensity distributions, and centroid image rendering (Matplotlib, Seaborn).
- Image matrix unpacking, normalization, and spatial reshaping ($28 \times 28$ grayscale pixels).
- Machine learning workflow stages: Data Ingestion $\to$ EDA $\to$ Preprocessing $\to$ Feature Selection $\to$ Splitting $\to$ Model Selection.

## Algorithms / Techniques

- **Modular Tabular EDA (`eda_tabular`)**: Automated extraction of dataset dimensions, column types, missingness counts, duplicate rows, descriptive metrics (`describe()`), numerical histograms, correlation heatmaps, and target class distributions.
- **Modular Image EDA (`eda_images`)**: Multidimensional image data diagnostics, class balance verification, random sample grid rendering, pixel intensity distribution analysis, and dataset-wide average image generation.
- **Feature Selection Mapping**:
  - Continuous features: ANOVA F-test / `SelectKBest`
  - Categorical & count features: Chi-Square test ($\chi^2$) / `SelectKBest`
  - High-dimensional image data: Dimensionality reduction via Principal Component Analysis (PCA)

## Datasets

The experiment explores five benchmark datasets:

1. **Loan Prediction Dataset** (`datasets/loan_prediction/train.csv`):
   - **Source:** Kaggle (Loan Prediction Problem Dataset)
   - **Samples & Features:** 614 instances, 13 attributes
   - **Target:** `Loan_Status` (Binary classification: Y / N)
   - **Characteristics:** 149 missing values distributed across 7 columns (`Gender`, `Married`, `Dependents`, `Self_Employed`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`).
2. **Pima Indians Diabetes Dataset** (`datasets/diabetes/diabetes.csv`):
   - **Source:** UCI Machine Learning Repository / Kaggle
   - **Samples & Features:** 768 instances, 9 numerical features
   - **Target:** `Outcome` (Binary classification: 0 = Non-Diabetic, 1 = Diabetic)
   - **Characteristics:** Zero explicit `NaN` values, but contains physiologically impossible 0-values in features such as `Insulin` and `SkinThickness` indicating encoded missing entries.
3. **MNIST Handwritten Digit Dataset** (`datasets/mnist/mnist_train.csv`):
   - **Source:** Kaggle (`mnist-in-csv`)
   - **Samples & Features:** 60,000 instances, 785 columns (784 pixel values + 1 label)
   - **Target:** `label` (Multi-class classification: digits $0$ through $9$)
   - **Characteristics:** Grayscale $28 \times 28$ pixel matrices flattened into 1D vectors with values in $[0, 255]$.
4. **Iris Species Dataset & SMS Spam Collection**:
   - Analyzed for formal ML task mapping, feature selection strategies, and algorithm recommendations.

## Implementation

The notebook ([ex1.ipynb](./ex1.ipynb)) implements two core analytical functions:

1. `eda_tabular(df, target=None)`:
   - Separates numerical and categorical variables.
   - Reports shape, dtypes, null counts, and duplicates.
   - Plots histograms for numeric features and computes a Pearson correlation matrix.
   - Produces frequency bar plots for categorical targets.
2. `eda_images(df, label_col='label', image_size=(28, 28), samples=9)`:
   - Separates pixel features from class labels.
   - Computes class balance across digit classes.
   - Displays a $3 \times 3$ grid of randomly selected digits with their ground-truth labels.
   - Generates a histogram of pixel intensities across all samples.
   - Computes and displays the global mean image ($\mu_{\text{pixel}}$) across all 60,000 instances.

## Results / Analysis

### Dataset Diagnostics Summary

| Dataset | ML Task | Missing Values | Class Balance / Distribution | Recommended ML Model |
|---|---|---|---|---|
| **Loan Prediction** | Binary Classification | 149 total across 7 features | Imbalanced loan approval ratio | Logistic Regression, Random Forest |
| **Diabetes** | Binary Classification | 0 `NaN` (implicit 0s in `Insulin`, `SkinThickness`) | Imbalanced (65.1% Class 0, 34.9% Class 1) | Logistic Regression, Naïve Bayes, KNN |
| **MNIST** | Multi-class Classification | 0 | Balanced (~5,400–6,700 samples/class) | KNN, CNN, Random Forest |
| **Iris** | Multi-class Classification | 0 | Perfectly balanced (50 samples/class) | Decision Tree, SVM, KNN |
| **SMS Spam** | Binary Text Classification | High sparsity in auxiliary cols | Imbalanced (86.6% Ham, 13.4% Spam) | Multinomial Naïve Bayes, Linear SVM |

### Key Observations
- **Clinical Feature Correlation:** In the Diabetes dataset, `Glucose` exhibits the highest positive correlation with `Outcome`, followed by `BMI` and `Age`, aligning with known clinical risk factors.
- **Image Pixel Sparsity:** In the MNIST dataset, pixel intensities are heavily concentrated at 0 (mean: 33.32, std: 78.57), indicating significant spatial background sparsity.
- **Digit Centering:** The average MNIST image displays a clear, centered circular/elliptical stroke pattern, confirming that the digits are consistently normalized and centered within the $28 \times 28$ bounding frame.

## Files

- [ex1.ipynb](./ex1.ipynb) — Jupyter notebook containing the implementation of `eda_tabular` and `eda_images`.
- [ex1.html](./ex1.html) — HTML export of the executed notebook with embedded visualizations.
- [Sharruk_S_3122247001061_Assignment1.pdf](./Sharruk_S_3122247001061_Assignment1.pdf) — Comprehensive laboratory report with theoretical background, figures, inferences, and task classification tables.
- [question/Experiment_1.pdf](./question/Experiment_1.pdf) — Official experiment question specification and guidelines.

## Conclusion

Experiment 1 established the foundational workflows for data inspection, distribution diagnostics, and visualization across tabular and raster image domains. Identifying dataset idiosyncrasies—such as implicit missing values in clinical features and background sparsity in pixel arrays—highlighted the necessity of rigorous exploratory data analysis prior to machine learning model development.
