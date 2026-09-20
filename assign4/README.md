# Experiment 4 — Binary Classification using Linear and Kernel-Based Models

## Overview

This experiment explores linear and kernel-based binary classification algorithms on the **Spambase** dataset. It evaluates **Logistic Regression** alongside **Support Vector Machines (SVM)** across four distinct kernel configurations (**Linear**, **Polynomial**, **Radial Basis Function (RBF)**, and **Sigmoid**). The investigation focuses on maximal margin separation, slack variable soft-margin tradeoffs ($C$), non-linear feature space mappings via the kernel trick, exhaustive hyperparameter optimization with `GridSearchCV`, and bias-variance generalization dynamics.

## Objectives

- Implement and benchmark baseline and regularized **Logistic Regression** for binary classification.
- Implement **Support Vector Classifiers (SVC)** and analyze the impact of different kernel transformations:
  - Linear Kernel
  - Polynomial Kernel ($\text{degree}=3$)
  - Radial Basis Function (RBF / Gaussian) Kernel
  - Sigmoid (Hyperbolic Tangent) Kernel
- Standardize the 57 continuous predictors using `StandardScaler` to ensure scale-invariant geometric margins.
- Optimize hyperparameters via 5-fold `GridSearchCV`:
  - Logistic Regression: Inverse regularization strength $C$, penalty ($L_1, L_2$), and solvers (`liblinear`, `lbfgs`).
  - SVM: Cost parameter $C$, kernel type, polynomial degree, and kernel coefficient $\gamma$.
- Compare models across test Accuracy, Precision, Recall, $F_1$-score, ROC-AUC, training latency, and train-test generalization gap.

## Concepts Covered

- Logistic regression log-odds hypothesis and cross-entropy loss:
  $$P(y=1 \mid x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$$
- Support Vector Machine primal soft-margin objective:
  $$\min_{w, b, \xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^n \xi_i \quad \text{s.t.} \quad y_i (w^T \phi(x_i) + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$
- The Kernel Trick: computing inner products in high-dimensional feature spaces without explicit transformation:
  - Linear: $K(x, x') = x^T x'$
  - Polynomial: $K(x, x') = (\gamma x^T x' + r)^d$
  - RBF: $K(x, x') = \exp(-\gamma \|x - x'\|^2)$
  - Sigmoid: $K(x, x') = \tanh(\gamma x^T x' + r)$
- Margin geometry: support vectors, hard vs. soft margins, and decision boundary curvature.
- Bias-variance tradeoff in kernel methods.

## Algorithms / Techniques

- **Classifiers**: `LogisticRegression`, `SVC`
- **Feature Preprocessing**: `StandardScaler` fitted strictly on training data
- **Hyperparameter Optimization**: 5-Fold Stratified `GridSearchCV`
- **Validation**: 80:20 Stratified Train-Test split and 5-Fold Cross-Validation

## Dataset

- **Dataset Name:** Spambase Dataset (`datasets/spambase.csv`)
- **Source:** UCI Machine Learning Repository / Kaggle
- **Number of Samples:** 4,601 instances (2,788 Ham / 60.6%, 1,813 Spam / 39.4%)
- **Number of Features:** 57 continuous attributes (word frequencies, character frequencies, capital run lengths)
- **Target:** `class` (0 = Legitimate / Ham, 1 = Spam)
- **Data Quality:** 0 missing values; 391 duplicate rows detected
- **Split:** 80% Train (3,680 instances), 20% Test (921 instances)

## Implementation

The notebook ([ex4.ipynb](./ex4.ipynb)) executes the following steps:

1. **Exploratory Data Analysis:** Summary statistics, class balance assessment, and a consolidated 12-subplot visual overview.
2. **Preprocessing:** Scales all 57 numerical predictors using `StandardScaler`.
3. **Baseline Logistic Regression:** Evaluates default $L_2$-penalized Logistic Regression with `lbfgs`.
4. **Tuned Logistic Regression:** Searches $C \in \{0.01, 0.1, 1, 10, 100\}$, penalty $\in \{L_1, L_2\}$, and solvers $\in \{\text{'liblinear'}, \text{'lbfgs'}\}$.
5. **SVM Kernel Comparison:** Trains un-tuned SVCs across Linear, Polynomial ($d=3$), RBF, and Sigmoid kernels to isolate kernel-induced inductive bias.
6. **Tuned SVM:** Executes a grid search over $C \in \{0.1, 1, 10, 100\}$, $\gamma \in \{\text{'scale'}, \text{'auto'}, 0.01, 0.1\}$, and kernels $\in \{\text{'rbf'}, \text{'poly'}\}$.
7. **Cross-Validation & Evaluation:** Compares 5-fold CV fold distributions and produces ROC-AUC curves, confusion matrices, and train-test gap tables.

## Results / Analysis

### 1. Hyperparameter Tuning Results (5-Fold CV)

| Model | Search Method | Optimal Parameters | Best CV Accuracy |
|---|---|---|---|
| **Logistic Regression** | `GridSearchCV` | `C = 100`, `penalty = 'l1'`, `solver = 'liblinear'` | 0.9266 |
| **SVM** | `GridSearchCV` | `C = 10`, `gamma = 'scale'`, `kernel = 'rbf'`, `degree = 2` | **0.9361** |

### 2. Performance Comparison on Held-Out Test Set

| Model | Test Accuracy | Precision | Recall | $F_1$-Score | Training Time (s) | ROC-AUC |
|---|---|---|---|---|---|---|
| **Baseline Logistic Regression** | **0.9294** | 0.9209 | **0.8981** | **0.9093** | **0.0123** | — |
| **SVM (Linear)** | **0.9294** | 0.9209 | **0.8981** | **0.9093** | 0.3431 | **0.9695** |
| **SVM (RBF)** | 0.9273 | 0.9277 | 0.8843 | 0.9055 | 0.1827 | 0.9667 |
| **Tuned Logistic Regression** | 0.9262 | 0.9198 | 0.8904 | 0.9048 | 0.4291 | — |
| **Tuned SVM (RBF, $C=10$)** | 0.9207 | 0.9143 | 0.8815 | 0.8976 | 0.1513 | — |
| **SVM (Sigmoid)** | 0.8849 | 0.8599 | 0.8457 | 0.8528 | 0.2408 | — |
| **SVM (Polynomial, $d=3$)** | 0.7796 | **0.9598** | 0.4601 | 0.6220 | 0.2933 | — |

### 3. Generalization & Bias-Variance Analysis

| Model | Train Accuracy | Test Accuracy | Generalization Gap ($\text{Train} - \text{Test}$) | Bias / Variance Diagnosis |
|---|---|---|---|---|
| **Baseline Logistic Regression** | 0.9307 | 0.9294 | **0.0013** | Low bias, exceptionally low variance |
| **SVM (Linear)** | 0.9329 | 0.9294 | 0.0035 | Low bias, low variance |
| **SVM (RBF)** | 0.9481 | 0.9273 | 0.0208 | Low bias, well-regularized variance |
| **Tuned SVM ($C=10$)** | 0.9701 | 0.9207 | **0.0494** | Low bias, mild overfitting to training noise |
| **SVM (Sigmoid)** | 0.8818 | 0.8849 | -0.0031 | Moderate bias (non-PSD kernel behavior) |
| **SVM (Polynomial)** | 0.7938 | 0.7796 | 0.0142 | High bias (poor recall on $d=3$ boundary) |

### Key Observations
- **Linear Decision Boundary Suitability:** Baseline Logistic Regression and Linear SVM tied for top test accuracy ($92.94\%$) and $F_1$-score ($0.9093$), indicating that word-frequency features in Spambase are largely linearly separable once scaled.
- **Computational Efficiency:** Baseline Logistic Regression completed training in $0.0123\text{ s}$—over **27× faster** than Linear SVM ($0.3431\text{ s}$) while matching its classification fidelity exactly.
- **Overfitting with Complex Kernels:** Increasing SVM complexity ($C=10$ with RBF) elevated training accuracy to $97.01\%$ but reduced test accuracy to $92.07\%$, confirming that higher margin penalties fit idiosyncratic training noise.
- **Polynomial Kernel Deficiency:** The un-tuned degree-3 polynomial kernel suffered from high bias, capturing only $46.01\%$ recall despite high precision ($95.98\%$).

## Files

- [ex4.ipynb](./ex4.ipynb) — Jupyter notebook containing the preprocessing, SVM kernel benchmarking, GridSearchCV tuning, and evaluation routines.
- [Experiment_4_Report_Sharruk_S_3122247001061.pdf](./Experiment_4_Report_Sharruk_S_3122247001061.pdf) — Complete laboratory report including mathematical formulations, slack variable theory, and comprehensive result tables.
- [question/Experiment_4.pdf](./question/Experiment_4.pdf) — Official experiment specification and question sheet.

## Conclusion

Experiment 4 demonstrated that complex non-linear kernel mappings do not always outperform simpler linear formulations. For the 57-dimensional Spambase problem, linear decision surfaces (Logistic Regression and Linear SVM) provided optimal generalization with minimal variance. Logistic Regression proved to be the most practical model, achieving the highest test accuracy ($92.94\%$) with negligible training latency ($0.0123\text{ s}$) and an exceptionally small generalization gap ($0.0013$).
