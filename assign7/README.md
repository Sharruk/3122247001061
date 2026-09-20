# Experiment 7 — Dimensionality Reduction and Model Evaluation (With and Without PCA)

## Overview

This experiment investigates the empirical effects of linear dimensionality reduction via **Principal Component Analysis (PCA)** on classifier performance across diverse machine learning paradigms. Using the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset, 30 continuous features are projected onto 10 orthogonal principal components (retaining **95.21%** of cumulative variance). Ten distinct classifiers—spanning linear, probabilistic, instance-based, tree-based, boosting, and stacking architectures—are trained, hyperparameter-tuned, and cross-validated under two paired regimes: **No-PCA** and **With-PCA**.

## Objectives

- Apply Principal Component Analysis (PCA) to compress high-dimensional feature spaces while preserving target variance ($\ge 95\%$).
- Implement, tune, and evaluate **10 distinct classifiers** under both **No-PCA** and **With-PCA** conditions:
  1. Support Vector Machine (SVM)
  2. Gaussian Naïve Bayes
  3. K-Nearest Neighbors (KNN)
  4. Logistic Regression
  5. Decision Tree
  6. Random Forest
  7. AdaBoost
  8. Gradient Boosting
  9. XGBoost
  10. Stacked Ensemble
- Optimize hyperparameters using 5-fold `GridSearchCV` and `RandomizedSearchCV` for each model in both regimes.
- Analyze how orthogonal feature rotation and variance compression differently affect linear margin models, axis-aligned tree splits, and probabilistic density estimators.
- Compare fold-wise cross-validation stability and held-out test performance.

## Concepts Covered

- **Principal Component Analysis (PCA)**:
  - Sample covariance matrix eigen-decomposition: $\Sigma = \frac{1}{n} X^T X = V \Lambda V^T$
  - Variance retention: $\frac{\sum_{i=1}^k \lambda_i}{\sum_{j=1}^d \lambda_j} \ge 0.95$
  - Orthogonal feature projection: $Z = X V_k$
- **Inductive Bias & Dimensionality**:
  - *Linear / Margin models (Logistic Regression, SVM)* benefit from feature decorrelation and collinearity removal.
  - *Tree-based models (Decision Trees, Random Forests, Boosting)* rely on axis-aligned splits along individual physical dimensions; linear combinations can dilute interpretable split thresholds.
  - *Probabilistic models (Gaussian NB)* assume class-conditional feature independence, which is affected by linear principal transformations.
- 5-Fold Cross-Validation stability (fold mean and standard deviation).

## Algorithms / Techniques

- **Dimensionality Reduction**: `PCA(n_components=10)` (retaining 95.21% cumulative variance)
- **Classifiers Evaluated (10 Models)**:
  - `SVC` (probability enabled)
  - `GaussianNB` (with `var_smoothing` tuning)
  - `KNeighborsClassifier` (distance & metric tuning)
  - `LogisticRegression` (with $L_1/L_2$ regularizers)
  - `DecisionTreeClassifier` (pruning tuning)
  - `RandomForestClassifier` (ensemble tuning)
  - `AdaBoostClassifier`
  - `GradientBoostingClassifier`
  - `XGBClassifier`
  - `StackingClassifier` (SVM + Naïve Bayes + Decision Tree $\to$ Logistic Regression)
- **Validation Pipeline**: 80:20 Stratified Train-Test split and 5-Fold Stratified Cross-Validation

## Dataset

- **Dataset Name:** Wisconsin Diagnostic Breast Cancer (WDBC) (`datasets/breastCancer/wdbc.data`)
- **Source:** UCI Machine Learning Repository
- **Number of Samples:** 569 instances (357 Benign / 62.7%, 212 Malignant / 37.3%)
- **Original Features:** 30 continuous predictors derived from FNA cell nuclei images
- **PCA Compressed Features:** 10 orthogonal principal components (3:1 compression ratio)
- **Target:** `diagnosis` (Malignant = 1, Benign = 0)
- **Split:** 80% Train (455 samples), 20% Test (114 samples)

## Implementation

The notebook ([ex7.ipynb](./ex7.ipynb)) carries out the following structured evaluation:

1. **Preprocessing & Standardization:** Standardizes features using `StandardScaler` to prevent high-magnitude features from dominating PCA eigenvalues.
2. **PCA Analysis:** Computes the scree spectrum and cumulative explained variance. Selects $k=10$ components, capturing $95.21\%$ of the total variance.
3. **Paired Model Training:** For all 10 classifiers:
   - Configures identical hyperparameter search spaces for both regimes.
   - Executes 5-fold cross-validation with `GridSearchCV` or `RandomizedSearchCV` on the training set.
   - Records per-fold CV scores, standard deviations, and optimal parameter configurations.
4. **Held-Out Test Set Evaluation:** Evaluates best estimators on the 114 test samples, computing Accuracy, Precision, Recall, $F_1$, ROC-AUC, and inference latency.
5. **Comparative Diagnostics:** Generates side-by-side performance bar charts, ROC curves, PR curves, confusion matrices, and fold-wise stability plots.

## Results / Analysis

### 1. PCA Variance Retention Spectrum

| Setting | Features / Components | Cumulative Explained Variance (%) |
|---|---|---|
| **No-PCA** | All 30 original features | 100.00% |
| **With-PCA** | **10 Principal Components** | **95.21%** |

### 2. Comprehensive Model Comparison (Held-Out Test Set: No-PCA vs. With-PCA)

| Classifier | Setting | Test Accuracy | Precision | Recall | $F_1$-Score | Test ROC-AUC | Mean CV Acc (%) | Std CV Acc (%) |
|---|---|---|---|---|---|---|---|---|
| **SVM** | No-PCA | 0.9737 | 1.0000 | 0.9286 | 0.9630 | 0.9944 | 97.58% | 1.48% |
| | **With-PCA** | **0.9825** | **1.0000** | **0.9524** | **0.9756** | 0.9944 | **97.80%** | **1.45%** |
| **Logistic Regression** | **No-PCA** | **0.9825** | **1.0000** | **0.9524** | **0.9756** | 0.9960 | 97.36% | 1.25% |
| | **With-PCA** | **0.9825** | **1.0000** | **0.9524** | **0.9756** | 0.9950 | 97.36% | 1.83% |
| **Stacking** | No-PCA | 0.9649 | 1.0000 | 0.9048 | 0.9500 | 0.9954 | 96.92% | 1.34% |
| | **With-PCA** | **0.9737** | **1.0000** | **0.9286** | **0.9630** | 0.9954 | 96.70% | 1.48% |
| **XGBoost** | No-PCA | 0.9737 | 1.0000 | 0.9286 | 0.9630 | 0.9921 | 96.26% | 1.23% |
| | With-PCA | 0.9737 | 0.9756 | 0.9524 | 0.9639 | 0.9858 | 95.82% | 2.11% |
| **Random Forest** | No-PCA | 0.9737 | 1.0000 | 0.9286 | 0.9630 | 0.9940 | 96.26% | 1.58% |
| | With-PCA | 0.9474 | 0.9500 | 0.9048 | 0.9268 | 0.9835 | 94.73% | 1.63% |
| **AdaBoost** | No-PCA | 0.9737 | 1.0000 | 0.9286 | 0.9630 | 0.9871 | 96.48% | 1.48% |
| | With-PCA | 0.9649 | 0.9750 | 0.9286 | 0.9512 | 0.9894 | 95.60% | 1.95% |
| **Gradient Boosting** | No-PCA | 0.9649 | 1.0000 | 0.9048 | 0.9500 | 0.9874 | 97.58% | 0.82% |
| | With-PCA | 0.9561 | 0.9744 | 0.9048 | 0.9383 | 0.9881 | 96.04% | 1.83% |
| **Decision Tree** | No-PCA | 0.9649 | 1.0000 | 0.9048 | 0.9500 | 0.9744 | 93.63% | 2.45% |
| | With-PCA | 0.9386 | 0.9487 | 0.8810 | 0.9136 | 0.9475 | 93.41% | 2.66% |
| **KNN** | No-PCA | 0.9649 | 1.0000 | 0.9048 | 0.9500 | 0.9868 | 96.92% | 1.63% |
| | With-PCA | 0.9474 | 0.9737 | 0.8810 | 0.9250 | 0.9845 | 96.04% | 1.15% |
| **Naive Bayes** | No-PCA | 0.9211 | 0.9231 | 0.8571 | 0.8889 | 0.9821 | 93.85% | 2.05% |
| | With-PCA | 0.8947 | 0.8571 | 0.8571 | 0.8571 | 0.9722 | 92.53% | 2.62% |

### 3. Key Observations & Hypotheses Verification

1. **Which models improved most with PCA?**
   - **Support Vector Machines (SVM)** achieved the largest accuracy increase ($+0.88\%$, rising from $97.37\%$ to **$98.25\%$**), and **Stacking** rose from $96.49\%$ to $97.37\%$. Margin-based classifiers benefit because PCA projects the data onto orthogonal axes that maximize variance and suppress noise in minor directions.
2. **Which models degraded under PCA?**
   - **Gaussian Naïve Bayes** deteriorated the most ($-2.64\%$, falling from $92.11\%$ to $89.47\%$). Although PCA decorrelates features linearly ($\text{Cov}(z_i, z_j) = 0$), it does not ensure mutual statistical independence for non-Gaussian distributions, leading to suboptimal density estimations.
   - **Decision Trees and Random Forests** dropped by $-2.63\%$ (from $96.49\%$ to $93.86\%$ for DT; $97.37\%$ to $94.74\%$ for RF). Decision trees split recursively along individual axes. Projecting original physiological attributes into dense linear combinations forces trees to approximate diagonal decision boundaries through deep, complex step functions, hurting generalization.
3. **Linear Models vs. Ensembles under PCA:**
   - Linear and margin models (Logistic Regression: $98.25\%$, SVM: $98.25\%$) outperformed or matched all non-linear ensembles under PCA. This demonstrates that once dimensionality is compressed into orthogonal axes of maximum variance, linear hyperplanes provide optimal, low-variance decision boundaries.

## Files

- [ex7.ipynb](./ex7.ipynb) — Jupyter notebook with complete implementation of the 10 models under No-PCA and With-PCA conditions.
- [Experiment_7_Report_Sharruk_S_3122247001061.pdf](./Experiment_7_Report_Sharruk_S_3122247001061.pdf) — Comprehensive laboratory report containing mathematical formulations of PCA, 10-model comparison tables, ROC curves, and detailed discussion responses.
- [question/Assignment_7_question.pdf](./question/Assignment_7_question.pdf) — Official experiment assignment specification.

## Conclusion

Experiment 7 revealed that dimensionality reduction is not uniformly beneficial across machine learning model families. For linear and margin-based classifiers (SVM, Logistic Regression), PCA compression successfully stripped collinear noise and improved or maintained peak classification accuracy ($98.25\%$). In contrast, for tree-based algorithms and probabilistic classifiers, projecting raw features into linear combinations disrupted natural axis-aligned splits and class-conditional distributions.
