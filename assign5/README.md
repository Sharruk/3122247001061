# Experiment 5 — Decision Tree and Random Forest: A Comparative Classification Study

## Overview

This experiment presents a comparative classification study between a single **Decision Tree** and a **Random Forest** ensemble on the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset. It evaluates the mechanisms of tree-based partitioning—specifically Information Gain (Entropy) versus Gini Impurity, tree pruning hyperparameters, bootstrap aggregation (bagging), and random feature subspace projection. The study analyzes how ensembling uncorrelated decision trees reduces model variance, stabilizes cross-validation predictions, and delivers superior probabilistic discrimination for clinical oncology diagnosis.

## Objectives

- Implement a single **Decision Tree Classifier** for benign versus malignant tumor classification.
- Extend individual decision trees into a **Random Forest Ensemble** using bagging and feature subsampling.
- Study the impact of tree pruning hyperparameters (`max_depth`, `min_samples_split`, `min_samples_leaf`) on tree complexity and overfitting.
- Perform systematic hyperparameter optimization using 5-fold cross-validation with `GridSearchCV`.
- Compare model performance across Accuracy, Precision, Recall, $F_1$-score, ROC curves, and Area Under the Curve (ROC-AUC).
- Extract and interpret **Gini Feature Importances** from the Random Forest ensemble to identify key clinical biomarkers.

## Concepts Covered

- Splitting criteria:
  - Entropy / Information Gain: $H(S) = -\sum_{i=1}^c p_i \log_2 p_i$
  - Gini Impurity: $I_G(p) = 1 - \sum_{i=1}^c p_i^2$
- Pre-pruning and complexity regularization: controlling maximum depth and minimum leaf samples to prevent leaf purity overfitting.
- Bootstrap Aggregation (Bagging): reducing prediction variance by averaging independently grown decorrelated trees.
- Random Feature Projection: choosing a random subset of $\sqrt{d}$ features at each split to decorrelate individual trees.
- Feature Importance Ranking: Mean Decrease in Impurity (MDI).

## Algorithms / Techniques

- **Tree Classifiers**: `DecisionTreeClassifier`, `RandomForestClassifier`
- **Hyperparameter Optimization**: Exhaustive 5-Fold Stratified `GridSearchCV`
- **Validation Scheme**: 80:20 Stratified Train-Test split and 5-Fold Cross-Validation
- **Diagnostics**: Confusion matrices, ROC-AUC curves, and feature importance bar plots

## Dataset

- **Dataset Name:** Wisconsin Diagnostic Breast Cancer (WDBC) (`datasets/breastCancer/wdbc.data`)
- **Source:** UCI Machine Learning Repository
- **Number of Samples:** 569 instances (357 Benign / 62.7%, 212 Malignant / 37.3%)
- **Number of Features:** 30 continuous real-valued features computed from fine needle aspirate (FNA) images:
  - Ten basic characteristics: *radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension*
  - Expressed across three statistics: Mean (`mean`), Standard Error (`se`), and Largest/Worst (`worst`)
- **Target:** `diagnosis` (M = Malignant / 1, B = Benign / 0)
- **Data Quality:** 0 missing values; patient ID attribute removed as non-predictive.
- **Split:** 80% Train (455 samples), 20% Test (114 samples)

## Implementation

The notebook ([ex5.ipynb](./ex5.ipynb)) executes the following steps:

1. **Exploratory Data Analysis:** Column type verification, class distribution inspection, feature correlation heatmaps, and a 12-plot visual summary.
2. **Data Preprocessing:** Encodes `diagnosis` to binary integers and drops the patient identifier.
3. **Baseline Models:** Evaluates unconstrained baseline Decision Tree and Random Forest classifiers.
4. **Decision Tree Hyperparameter Tuning:** Explores a 90-combination grid over:
   - `criterion`: `['gini', 'entropy']`
   - `max_depth`: `[3, 5, 7, 10, None]`
   - `min_samples_split`: `[2, 5, 10]`
   - `min_samples_leaf`: `[1, 2, 4]`
5. **Random Forest Hyperparameter Tuning:** Explores an exhaustive grid over:
   - `n_estimators`: `[50, 100, 200]`
   - `max_depth`: `[5, 10, 15, None]`
   - `max_features`: `['sqrt', 'log2']`
   - `bootstrap`: `[True, False]`
6. **Cross-Validation Comparison:** Evaluates per-fold accuracy stability across 5 folds.
7. **Final Evaluation:** Evaluates both tuned models on the held-out test set and plots ROC curves, confusion matrices, and feature importances.

## Results / Analysis

### 1. Optimal Hyperparameters from 5-Fold Cross-Validation

| Model | Optimal Parameters | Mean CV Accuracy | Mean CV $F_1$-Score | Tuning Latency |
|---|---|---|---|---|
| **Decision Tree** | `criterion='entropy'`, `max_depth=5`, `min_samples_leaf=1`, `min_samples_split=2` | 93.63% | 0.9121 | 4.5 s |
| **Random Forest** | `n_estimators=50`, `max_depth=10`, `max_features='sqrt'`, `bootstrap=False` | **97.14%** | **0.9617** | 33.3 s |

### 2. 5-Fold Cross-Validation Accuracy Comparison

| Fold | Decision Tree Accuracy | Random Forest Accuracy |
|---|---|---|
| Fold 1 | 0.9011 | **0.9670** |
| Fold 2 | 0.9560 | **0.9890** |
| Fold 3 | 0.9121 | **0.9670** |
| Fold 4 | 0.9560 | **0.9560** |
| Fold 5 | 0.9560 | **0.9780** |
| **Average CV Accuracy** | **93.63%** | **97.14%** |

*Inference:* Random Forest outperforms the Decision Tree across every individual validation fold, achieving a $+3.51\%$ higher mean accuracy with significantly tighter fold-to-fold consistency.

### 3. Held-Out Test Set Performance (114 Instances)

| Model | Test Accuracy | Precision | Recall | $F_1$-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Baseline Decision Tree** | 0.9298 | 0.9048 | 0.9048 | 0.9048 | 0.9246 |
| **Baseline Random Forest** | **0.9737** | **1.0000** | **0.9286** | **0.9630** | 0.9929 |
| **Tuned Decision Tree** | 0.9649 | **1.0000** | 0.9048 | 0.9500 | 0.9744 |
| **Tuned Random Forest** | 0.9649 | **1.0000** | 0.9048 | 0.9500 | **0.9950** |

### 4. Clinical Significance & ROC-AUC Discrimination

- **High Precision:** Both tuned models achieved **100% precision** (Precision = 1.0000) on the test set, generating zero false positives (no benign cases were incorrectly classified as malignant).
- **Discrimination Superiority:** Although the discrete threshold predictions for the tuned DT and RF matched on the 114 test samples, Random Forest demonstrated vastly superior continuous ranking ability, achieving an **ROC-AUC of 0.9950** compared to **0.9744** for the Decision Tree.
- **Top Biomarkers:** Feature importance analysis revealed that extreme tumor morphology features—specifically `worst perimeter`, `worst concave points`, `worst radius`, and `mean concave points`—provide the highest discriminative signal for malignancy detection.

## Files

- [ex5.ipynb](./ex5.ipynb) — Jupyter notebook containing the full tree tuning pipeline, cross-validation, and visualization routines.
- [Experiment_5_Report_Sharruk_S_3122247001061.pdf](./Experiment_5_Report_Sharruk_S_3122247001061.pdf) — Complete laboratory report including tree theory, entropy equations, bagging mathematics, and analysis questions.
- [question/Experiment_5.pdf](./question/Experiment_5.pdf) — Official experiment specification and question sheet.

## Conclusion

Experiment 5 highlighted the power of ensemble learning over single decision trees. Pruning a single decision tree improved its test accuracy from $92.98\%$ to $96.49\%$, but the model remained vulnerable to structural variance. Random Forest resolved this vulnerability by averaging decorrelated trees, boosting cross-validation accuracy to $97.14\%$ and achieving near-perfect probabilistic discrimination (ROC-AUC: $0.9950$).
