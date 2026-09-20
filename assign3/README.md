# Experiment 3 — Regression Analysis using Linear and Regularized Models

## Overview

This experiment explores linear and penalized regression techniques to predict a continuous financial target—the sanctioned loan amount (`LoanAmount`)—from the **Loan Prediction** dataset. It investigates the limitations of unconstrained Ordinary Least Squares (OLS) regression when dealing with collinear and noisy predictors, and demonstrates how shrinkage methods—**Ridge ($L_2$)**, **Lasso ($L_1$)**, and **Elastic Net ($L_1 + L_2$)**—mitigate overfitting, stabilize coefficient estimates, and improve out-of-sample generalization.

## Objectives

- Formulate and implement an end-to-end regression pipeline using Scikit-Learn `Pipeline` and `ColumnTransformer`.
- Train and evaluate baseline **Ordinary Least Squares (OLS) Linear Regression**.
- Implement and optimize regularized regression models:
  - **Ridge Regression** ($L_2$ Tikhonov regularization)
  - **Lasso Regression** ($L_1$ penalty inducing parameter sparsity)
  - **Elastic Net Regression** (convex combination of $L_1$ and $L_2$ penalties)
- Tune regularization hyperparameters ($\alpha$ and $\text{l1\_ratio}$) using 5-fold cross-validation with $R^2$ scoring.
- Perform residual diagnostics, learning curve analysis, and coefficient path comparisons.
- Quantify the bias-variance tradeoff across models using train-test divergence and fold-wise stability metrics.

## Concepts Covered

- Multiple Linear Regression objective formulation:
  $$\min_{w} \frac{1}{2n} \|Xw - y\|_2^2$$
- Ridge penalty ($L_2$) for multicollinearity dampening:
  $$\min_{w} \frac{1}{2n} \|Xw - y\|_2^2 + \alpha \|w\|_2^2$$
- Lasso penalty ($L_1$) for continuous feature shrinkage and variable selection:
  $$\min_{w} \frac{1}{2n} \|Xw - y\|_2^2 + \alpha \|w\|_1$$
- Elastic Net formulation balancing grouping and sparsity:
  $$\min_{w} \frac{1}{2n} \|Xw - y\|_2^2 + \alpha \left( \rho \|w\|_1 + \frac{1-\rho}{2} \|w\|_2^2 \right)$$
- Evaluation metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Coefficient of Determination ($R^2$).
- Residual analysis: homoscedasticity vs. heteroscedasticity, residual normality, and systematic prediction error.

## Algorithms / Techniques

- **Linear Regressors**: `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`
- **Data Transformation Pipeline**:
  - Continuous Features: Median imputation via `SimpleImputer` + `StandardScaler`
  - Categorical Features: Most frequent imputation + `OneHotEncoder(drop='first')`
- **Hyperparameter Optimization**: Exhaustive 5-fold `GridSearchCV` scoring on $R^2$
- **Diagnostics**: Learning curves (`learning_curve`), residual plots, and coefficient shrinkage tracking

## Dataset

- **Dataset Name:** Loan Prediction Dataset (`datasets/loan_prediction/train.csv`)
- **Target Variable:** `LoanAmount` (Continuous: sanctioned loan amount in thousands of dollars)
- **Target Handling:** Instances with missing target labels are dropped to prevent label leakage.
- **Predictors:**
  - *Numerical:* `ApplicantIncome`, `CoapplicantIncome`, `Loan_Amount_Term`, `Credit_History`
  - *Categorical:* `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, `Property_Area`, `Loan_Status`
  - *Identifier:* `Loan_ID` (dropped as non-predictive)
- **Split:** 80% Training, 20% Testing (`random_state=42`)

## Implementation

The notebook ([ex3.ipynb](./ex3.ipynb)) implements the following structured workflow:

1. **Exploratory Data Analysis:** Computes summary statistics, skewness, missing value counts, and generates a consolidated 12-subplot visual summary including target distribution and scatter plots.
2. **Preprocessing Pipeline:** Encapsulates imputation, feature scaling, and one-hot encoding into a unified `ColumnTransformer`.
3. **Baseline Fitting:** Evaluates unregularized and default penalized models on the train-test split.
4. **Hyperparameter Grid Search:**
   - Ridge: Searches $\alpha \in \{0.001, 0.01, 0.1, 1, 10, 100, 1000\}$
   - Lasso: Searches $\alpha \in \{0.001, 0.01, 0.1, 1, 10, 100, 1000\}$
   - Elastic Net: Searches $\alpha \in \{0.001, 0.01, 0.1, 1, 10, 100\}$ and $\text{l1\_ratio} \in \{0.1, 0.3, 0.5, 0.7, 0.9\}$
5. **Cross-Validation Benchmarks:** Measures per-fold $R^2$ and RMSE across 5 cross-validation folds.
6. **Diagnostics & Visualizations:** Generates predicted vs. actual plots, residual distribution plots, learning curves, and feature coefficient bar charts.

## Results / Analysis

### 1. Hyperparameter Optimization Summary (5-Fold CV)

| Model | Search Method | Best Hyperparameters | Best CV $R^2$ | Refit Latency (s) |
|---|---|---|---|---|
| **Ridge Regression** | `GridSearchCV` | `alpha = 100` | 0.2795 | 0.0092 |
| **Lasso Regression** | `GridSearchCV` | `alpha = 10` | 0.2209 | 0.0084 |
| **Elastic Net Regression** | `GridSearchCV` | `alpha = 1`, `l1_ratio = 0.5` | **0.2895** | 0.0086 |

### 2. 5-Fold Cross-Validation Performance (Average Across Folds)

| Model | MAE | MSE | RMSE | Mean CV $R^2$ | Fold $R^2$ Std |
|---|---|---|---|---|---|
| **Elastic Net Regression** | 47.3500 | **5800.86** | **75.5480** | **0.2895** | **0.0972** |
| **Ridge Regression** | **46.1167** | 5959.86 | 75.7788 | 0.2795 | 0.2082 |
| **Lasso Regression** | 48.6389 | 6480.74 | 78.5260 | 0.2209 | 0.2851 |
| **Linear Regression** | 45.9546 | 6926.32 | 79.0253 | 0.1840 | 0.4881 |

*Inference:* OLS Linear Regression suffered extreme instability in Fold 2 ($R^2 = -0.7829$), driving its fold variance up ($\sigma = 0.4881$). Elastic Net proved to be the most robust model, yielding the lowest standard deviation ($\sigma = 0.0972$) and highest mean CV $R^2$.

### 3. Final Test Set Evaluation (Held-Out Test Data)

| Model | Test MAE | Test MSE | Test RMSE | Test $R^2$ | Generalization Gap ($\text{Train } R^2 - \text{Test } R^2$) |
|---|---|---|---|---|---|
| **Elastic Net Regression** | 37.8258 | **3048.69** | **55.2149** | **0.1730** | **0.2199** |
| **Ridge Regression** | **36.7318** | 3092.85 | 55.6134 | 0.1610 | 0.2784 |
| **Lasso Regression** | 37.7971 | 3181.25 | 56.4026 | 0.1370 | 0.2525 |
| **Linear Regression** | 36.9875 | 3339.92 | 57.7921 | 0.0940 | 0.3627 |

### 4. Overfitting & Bias-Variance Analysis

- **OLS Overfitting:** Unregularized Linear Regression achieved a training $R^2$ of $0.4567$ but degraded to $0.0940$ on the test set, creating the largest generalization gap ($0.3627$).
- **Regularization Benefit:** Elastic Net constrained the model weights, achieving a training $R^2$ of $0.3929$ and test $R^2$ of $0.1730$, effectively reducing the generalization gap to $0.2199$ and improving test RMSE by $2.58$ units over OLS.
- **Lasso Sparsity:** At $\alpha = 10$, Lasso successfully shrank non-informative categorical coefficients to zero, verifying its intrinsic feature-selection capability.

## Files

- [ex3.ipynb](./ex3.ipynb) — Jupyter notebook containing the full preprocessing, hyperparameter search, cross-validation, and diagnostic plotting code.
- [Experiment_3_Report_Sharruk_S_3122247001061.pdf](./Experiment_3_Report_Sharruk_S_3122247001061.pdf) — Complete laboratory report containing theoretical background, mathematical derivations, tables, and discussions.
- [question/Experiment_3.pdf](./question/Experiment_3.pdf) — Laboratory experiment question sheet and problem specifications.

## Conclusion

Experiment 3 demonstrated that standard Ordinary Least Squares regression is prone to substantial variance and out-of-sample degradation when applied to real-world datasets with correlated features. Regularization via Elastic Net achieved the best overall performance by combining the weight-stabilizing effect of $L_2$ regularization with the sparsity benefits of $L_1$ regularization, resulting in the highest test $R^2$ (0.1730) and the smallest generalization gap.
