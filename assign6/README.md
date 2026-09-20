# Experiment 6 — Bagging, Boosting, and Stacked Ensemble Models

## Overview

This experiment conducts a rigorous comparative study of three foundational ensembling paradigms—**Bootstrap Aggregation (Bagging)**, **Boosting** (AdaBoost and Gradient Boosting), and **Stacking** (heterogeneous stacking with a meta-classifier)—applied to the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset. It explores how distinct ensemble mechanisms target and reduce the constituent components of generalization error (bias versus variance), benchmarks 5-fold cross-validation stability, and analyzes out-of-sample clinical classification fidelity.

## Objectives

- Implement and compare the three main families of ensemble learning:
  - **Bagging**: Parallel variance reduction via bootstrap aggregation and feature subspace sampling.
  - **Boosting**: Sequential bias reduction via **AdaBoost** (instance re-weighting) and **Gradient Tree Boosting** (gradient-directed pseudo-residual optimization).
  - **Stacking**: Heterogeneous architectural blending utilizing out-of-fold base learner predictions fed into a linear meta-learner.
- Systematically optimize ensemble hyperparameters using 5-fold cross-validation.
- Benchmark fold-to-fold cross-validation stability by computing empirical mean and standard deviation across validation iterations.
- Evaluate test set performance across Accuracy, Precision, Recall, $F_1$-score, ROC-AUC, and computational training latency.
- Conduct a formal bias-variance decomposition analysis to explain observed empirical gains.

## Concepts Covered

- **Bagging Principle**:
  $$f_{\text{bag}}(x) = \frac{1}{B} \sum_{b=1}^B f_b(x), \quad \text{Var}(f_{\text{bag}}) = \rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$$
  Reduces variance by averaging decorrelated estimators without increasing individual bias.
- **AdaBoost (Adaptive Boosting)**: Iteratively adjusts sample distribution weights $w_i^{(m)}$ based on exponential loss, focusing subsequent estimators on previously misclassified instances.
- **Gradient Boosting**: Iteratively fits weak decision trees to the negative gradient (pseudo-residuals) of the cross-entropy loss function:
  $$r_{im} = -\left[\frac{\partial L(y_i, f(x_i))}{\partial f(x_i)}\right]_{f(x) = f_{m-1}(x)}$$
- **Stacked Generalization**: Training heterogeneous base models (Support Vector Classifier, Gaussian Naïve Bayes, Decision Tree) and combining their predicted probability vectors using an $L_2$-regularized Logistic Regression meta-classifier.
- Bias-Variance tradeoff: Bagging targets high-variance estimators; Boosting targets high-bias estimators; Stacking targets structural hypothesis bias.

## Algorithms / Techniques

- **Baseline Model**: `DecisionTreeClassifier`
- **Bagging Model**: `BaggingClassifier` (with Decision Tree base estimators)
- **Boosting Models**: `AdaBoostClassifier`, `GradientBoostingClassifier`
- **Stacked Ensemble**: `StackingClassifier`
  - Base Learners: `SVC(probability=True)`, `GaussianNB()`, `DecisionTreeClassifier()`
  - Meta-Learner: `LogisticRegression(max_iter=1000)`
- **Validation**: 80:20 Stratified Train-Test split, 5-Fold Cross-Validation, and stability profiling

## Dataset

- **Dataset Name:** Wisconsin Diagnostic Breast Cancer (WDBC) (`datasets/breastCancer/wdbc.data`)
- **Source:** UCI Machine Learning Repository
- **Number of Samples:** 569 instances (357 Benign / 62.7%, 212 Malignant / 37.3%)
- **Number of Features:** 30 continuous real-valued cellular morphology predictors
- **Target:** `diagnosis` (Malignant = 1, Benign = 0)
- **Data Quality:** 0 missing values; ID attribute discarded
- **Split:** 80% Train (455 samples), 20% Test (114 samples)

## Implementation

The notebook ([ex6.ipynb](./ex6.ipynb)) executes the following pipeline:

1. **Exploratory Data Analysis:** Summary statistics, class balance assessment, feature correlation matrix, and 12-plot visual summary.
2. **Preprocessing:** Target encoding ($M \to 1, B \to 0$) and pipeline-based scaling (`StandardScaler`) where appropriate.
3. **Bagging Optimization:** Evaluates combinations over `n_estimators` $\in [50, 100, 200]$, `max_samples` $\in [0.5, 0.7, 1.0]$, `max_features` $\in [0.5, 0.7, 1.0]$, and `bootstrap` $\in [\text{True}, \text{False}]$.
4. **Boosting Optimization:**
   - AdaBoost: Tunes `n_estimators` $\in [50, 100, 200]$ and `learning_rate` $\in [0.01, 0.1, 1.0]$.
   - Gradient Boosting: Tunes `n_estimators` $\in [50, 100, 200]$, `learning_rate` $\in [0.01, 0.05, 0.1]$, and `max_depth` $\in [3, 5, 7]$.
5. **Stacking Implementation:** Assembles an SVC + Naïve Bayes + Decision Tree ensemble with a Logistic Regression meta-learner using 5-fold cross-validated out-of-fold prediction blending.
6. **Stability Analysis:** Measures fold-by-fold accuracy and $F_1$ variance to evaluate stability across all 5 folds.
7. **Final Evaluation:** Evaluates held-out test data, plotting ROC-AUC curves, confusion matrices, and bias-variance diagnostics.

## Results / Analysis

### 1. Hyperparameter Optimization & Model Selection (5-Fold CV)

| Model Family | Best Hyperparameters | Mean CV Accuracy | Mean CV $F_1$-Score |
|---|---|---|---|
| **Bagging** | `n_estimators=100`, `max_samples=0.7`, `max_features=0.7`, `bootstrap=True` | 97.143% | 96.214% |
| **AdaBoost** | `n_estimators=200`, `learning_rate=0.1` | 97.143% | 96.099% |
| **Gradient Boosting** | `n_estimators=200`, `learning_rate=0.1`, `max_depth=3` | **97.582%** | **96.742%** |
| **Stacked Ensemble** | Base: SVC + GaussianNB + DT; Meta: Logistic Regression | 97.143% | 96.096% |

*Selection:* **Gradient Boosting** was selected as the optimal boosting model due to its superior CV accuracy ($97.582\%$) and higher $F_1$-score ($96.742\%$).

### 2. 5-Fold Cross-Validation Stability Analysis

| Model | Mean CV Accuracy (%) | Std CV Accuracy (%) | Mean CV $F_1$ (%) | Std CV $F_1$ (%) |
|---|---|---|---|---|
| **Gradient Boosting** | **97.582%** | **0.822%** | **96.742%** | **1.090%** |
| **Stacked Ensemble** | 97.143% | 0.879% | 96.096% | 1.351% |
| **Bagging** | 97.143% | 2.153% | 96.214% | 2.798% |

*Inference:* Gradient Boosting displayed the greatest predictive stability, exhibiting a fold-to-fold accuracy standard deviation of just **0.822%**, compared to **2.153%** for Bagging.

### 3. Held-Out Test Set Performance (114 Instances)

| Model | Test Accuracy (%) | Precision | Recall | $F_1$-Score | ROC-AUC | Training Time (s) |
|---|---|---|---|---|---|---|
| **Bagging** | **97.37%** | **1.0000** | **0.9286** | **0.9630** | 0.9888 | 0.1160 |
| **Gradient Boosting** | **97.37%** | **1.0000** | **0.9286** | **0.9630** | 0.9874 | 0.4210 |
| **Stacked Ensemble** | **97.37%** | **1.0000** | **0.9286** | **0.9630** | **0.9980** | 0.1522 |

### 4. Key Observations & Comparison

- **Identical Discrete Test Predictions:** All three final ensembles achieved identical discrete accuracy ($97.37\%$) and $F_1$-score ($0.9630$) on the 114 test samples, misclassifying the exact same 3 malignant cases as benign with **zero false positives** (Precision = 1.0000).
- **ROC-AUC & Probability Calibration:** When evaluating continuous predicted probabilities across all classification thresholds, the **Stacked Ensemble** achieved the highest discrimination with an **ROC-AUC of 0.9980**, outperforming Bagging ($0.9888$) and Gradient Boosting ($0.9874$).
- **Complementary Base Learners:** In Stacking, the diverse decision boundaries of SVM (margin-based), Naïve Bayes (probabilistic), and Decision Tree (orthogonal splitting) enabled the Logistic Regression meta-learner to filter individual model uncertainties effectively.

## Files

- [ex6.ipynb](./ex6.ipynb) — Jupyter notebook containing the full ensemble training, tuning, stability calculations, and visualization code.
- [Experiment_6_Report_Sharruk_S_3122247001061.pdf](./Experiment_6_Report_Sharruk_S_3122247001061.pdf) — Laboratory report containing ensemble mathematics, stability derivations, ROC curves, and detailed discussion questions.
- [question/Experiment_6.pdf](./question/Experiment_6.pdf) — Official experiment specification and question sheet.

## Conclusion

Experiment 6 demonstrated the unique strengths of different ensembling paradigms. While Bagging effectively reduced variance and Stacking delivered the highest probability calibration (ROC-AUC: $0.9980$), **Gradient Boosting** proved to be the most dependable model overall, achieving the highest mean cross-validation accuracy ($97.58\%$) and the lowest fold-to-fold variance ($\sigma = 0.822\%$).
