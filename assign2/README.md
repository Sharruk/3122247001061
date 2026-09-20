# Experiment 2 — Email Spam/Ham Classification using Naïve Bayes and KNN

## Overview

This experiment explores probabilistic and instance-based machine learning algorithms on the benchmark **Spambase** dataset. The goal is to construct and evaluate spam detection models using three variants of **Naïve Bayes** (Gaussian, Multinomial, and Bernoulli) and **K-Nearest Neighbors (KNN)**. It conducts an in-depth empirical comparison of spatial indexing algorithms (`kd_tree`, `ball_tree`, `brute`), evaluates hyperparameter tuning via `GridSearchCV` versus `RandomizedSearchCV`, analyzes theoretical versus observed computational complexity, and validates generalizability through 5-fold cross-validation.

## Objectives

- Build and evaluate spam detection classifiers using Naïve Bayes and KNN.
- Compare the mathematical assumptions and empirical performance of **Gaussian**, **Multinomial**, and **Bernoulli Naïve Bayes**.
- Analyze the effect of varying the neighborhood size $k$ ($k=1, \dots, 10$) on classification accuracy and bias-variance tradeoff.
- Benchmark tree-based search structures (**KDTree**, **BallTree**) against **Brute-Force** distance calculation in a high-dimensional feature space ($d=57$).
- Optimize KNN hyperparameters using exhaustive `GridSearchCV` and budgeted `RandomizedSearchCV`.
- Measure training and inference execution times to compare theoretical time complexity against real-world execution.
- Validate classification stability using 5-fold cross-validation.

## Concepts Covered

- Probabilistic classification and Bayes' theorem with conditional independence assumptions:
  $$P(C \mid X) = \frac{P(X \mid C) P(C)}{P(X)}$$
- Distributional priors: Gaussian (continuous normal distribution), Multinomial (word frequency / count distributions), and Bernoulli (binary word presence / absence).
- Non-parametric lazy learning and majority voting in K-Nearest Neighbors.
- Distance metrics: Euclidean ($L_2$) versus Manhattan ($L_1$) distances, and distance-weighted voting ($w_i = 1 / d_i$).
- The curse of dimensionality in high-dimensional spatial indexing ($d=57$).
- Evaluation metrics: Accuracy, Precision, Recall, $F_1$-score, ROC-AUC, and Confusion Matrices.

## Algorithms / Techniques

- **Naïve Bayes Classifiers**: `GaussianNB`, `MultinomialNB`, `BernoulliNB`
- **K-Nearest Neighbors Classifier**: `KNeighborsClassifier`
- **Spatial Indexing Algorithms**: `auto`, `ball_tree`, `kd_tree`, `brute`
- **Hyperparameter Optimization**:
  - `GridSearchCV` (exhaustive grid exploration)
  - `RandomizedSearchCV` (randomized hyperparameter sampling)
- **Model Validation**: 80:20 Train-Test split and 5-Fold Stratified Cross-Validation

## Dataset

- **Dataset Name:** Spambase Dataset (`datasets/spambase.csv`)
- **Source:** UCI Machine Learning Repository / Kaggle
- **Number of Samples:** 4,601 instances (2,788 Ham / 60.6%, 1,813 Spam / 39.4%)
- **Number of Features:** 57 continuous predictors
  - 48 word-frequency features (`word_freq_*`)
  - 6 character-frequency features (`char_freq_*`)
  - 3 capital run-length metrics (average, longest, total)
- **Target:** `class` (0 = Ham / Legitimate, 1 = Spam)
- **Data Quality:** 0 missing values across all features; 391 duplicate rows detected.
- **Split:** 80% Training (3,680 samples), 20% Testing (921 samples).

## Implementation

The notebook ([ex2.ipynb](./ex2.ipynb)) carries out the following pipeline:

1. **Exploratory Data Analysis:** Summary statistics, missing value check, duplicate detection, class imbalance assessment, and correlation analysis.
2. **Naïve Bayes Modeling:** Trains Gaussian, Multinomial, and Bernoulli variants on the raw and normalized feature sets, evaluating test set metrics and training/prediction latency.
3. **KNN Neighborhood Analysis:** Iterates $k$ from 1 to 10 to trace test accuracy, precision, recall, and ROC-AUC curves to identify the empirical sweet spot ($k=5$).
4. **Hyperparameter Tuning:**
   - Evaluates a full grid of $k \in [1, \dots, 20]$, distance metrics (`euclidean`, `manhattan`), and weighting schemes (`uniform`, `distance`).
   - Compares the parameter recommendations, CV scores, and search times of `GridSearchCV` vs. `RandomizedSearchCV`.
5. **Spatial Indexing Benchmarks:** Benchmarks training latency and query prediction time across `auto`, `kd_tree`, `ball_tree`, and `brute`.
6. **Cross-Validation & Plots:** Performs 5-fold cross-validation across all models and generates confusion matrices, ROC curves, and precision-recall curves.

## Results / Analysis

### 1. Naïve Bayes Variants Comparison (Held-out Test Set)

| Model | Accuracy | Precision | Recall | $F_1$-Score | ROC-AUC | Training Time (s) | Prediction Time (s) |
|---|---|---|---|---|---|---|---|
| **Multinomial NB** | **0.8936** | **0.9431** | 0.7769 | **0.8520** | **0.9625** | 0.0050 | 0.0006 |
| **Bernoulli NB** | 0.8795 | 0.8684 | 0.8182 | 0.8426 | 0.9457 | 0.0119 | 0.0030 |
| **Gaussian NB** | 0.8328 | 0.7146 | **0.9587** | 0.8188 | 0.9229 | 0.0124 | 0.0023 |

*Inference:* Multinomial NB achieves the highest overall accuracy (89.36%) and ROC-AUC (0.9625) because word and character frequencies reflect token occurrence rates better than a normal distribution. Gaussian NB achieves extremely high recall (95.87%) at the expense of precision (71.46%), resulting in many false positives.

### 2. KNN Neighborhood Performance ($k=1 \dots 10$)

| $k$ | Accuracy | Precision | Recall | $F_1$-Score | ROC-AUC | Prediction Time (s) |
|---|---|---|---|---|---|---|
| $k=1$ | 0.8958 | 0.8678 | 0.8678 | 0.8678 | 0.8909 | 0.0715 |
| $k=2$ | 0.8936 | 0.9316 | 0.7879 | 0.8537 | 0.9228 | 0.0386 |
| $k=3$ | 0.8958 | 0.8825 | 0.8485 | 0.8652 | 0.9314 | 0.0531 |
| $k=4$ | 0.8903 | 0.8994 | 0.8127 | 0.8538 | 0.9367 | 0.0539 |
| **$k=5$ (Optimal Default)** | **0.9023** | **0.8760** | **0.8760** | **0.8760** | **0.9419** | **0.0513** |
| $k=6$ | 0.8969 | 0.8988 | 0.8320 | 0.8641 | 0.9442 | 0.0394 |
| $k=7$ | 0.8958 | 0.8761 | 0.8567 | 0.8663 | 0.9464 | 0.0427 |
| $k=8$ | 0.8958 | 0.9033 | 0.8237 | 0.8617 | 0.9490 | 0.0433 |
| $k=9$ | 0.8947 | 0.8866 | 0.8402 | 0.8628 | 0.9501 | 0.0719 |
| $k=10$ | 0.8882 | 0.8916 | 0.8154 | 0.8518 | 0.9514 | 0.0775 |

### 3. Hyperparameter Tuning Comparison

| Method | Best Hyperparameters | Best CV Accuracy | Test Accuracy | Test $F_1$ | Execution Time |
|---|---|---|---|---|---|
| **GridSearchCV** | `metric='manhattan'`, `n_neighbors=8`, `weights='distance'` | **0.9196** | **0.9142** | **0.8853** | 32.87 s |
| **RandomizedSearchCV** | `metric='manhattan'`, `n_neighbors=5`, `weights='distance'` | 0.9158 | 0.9131 | 0.8864 | **6.44 s** |

*Inference:* Both search methods discover that **Manhattan distance** with **distance weighting** outperforms standard unweighted Euclidean distance. `RandomizedSearchCV` converges to within $0.11\%$ of the exhaustive grid score while executing **5.1× faster**.

### 4. Spatial Indexing Complexity Benchmark

| Algorithm | Theoretical Train Time | Theoretical Query Time | Observed Train Time (s) | Observed Prediction Time (s) | Test Accuracy |
|---|---|---|---|---|---|
| **`brute`** | $\mathcal{O}(1)$ | $\mathcal{O}(n \cdot d)$ | **0.0019** | **0.0403** | 0.9023 |
| **`auto`** | — | — | 0.0040 | 0.0815 | 0.9023 |
| **`kd_tree`** | $\mathcal{O}(d \cdot n \log n)$ | $\mathcal{O}(2^d \log n)$ | 0.0343 | 0.4653 | 0.9023 |
| **`ball_tree`** | $\mathcal{O}(d \cdot n \log n)$ | $\mathcal{O}(d \log n)$ | 0.0293 | 0.5796 | 0.9023 |

*Inference:* Although KDTree and BallTree have lower theoretical query bounds in low dimensions, they succumb to the **curse of dimensionality** when $d=57$. In high dimensions, tree partition hyperspheres overlap substantially, forcing recursive search to inspect almost every leaf. Consequently, brute-force search with vectorized matrix operations runs **11.5× to 14.4× faster** during inference.

### 5. 5-Fold Cross-Validation Stability

| Model | Mean CV Accuracy | Standard Deviation |
|---|---|---|
| **Bernoulli NB** | **0.8809** | 0.0531 |
| **Multinomial NB** | 0.8687 | **0.0245** |
| **KNN (Default, $k=5$)** | 0.8646 | 0.0573 |
| **Gaussian NB** | 0.8222 | 0.0637 |

## Files

- [ex2.ipynb](./ex2.ipynb) — Jupyter notebook containing the complete implementation, benchmarks, and tuning routines.
- [ex2.html](./ex2.html) — HTML export of the executed notebook.
- [ICS1512_Exp2_SpamHam_Sharruk_S_3122247001061.pdf](./ICS1512_Exp2_SpamHam_Sharruk_S_3122247001061.pdf) — Comprehensive laboratory report containing detailed theoretical analysis, complexity tables, and discussion questions.
- [images/knn_confusion_matrix_k_5.png](./images/knn_confusion_matrix_k_5.png) — Confusion matrix plot generated for the optimal baseline KNN model ($k=5$).
- [question/Experiment_2.pdf](./question/Experiment_2.pdf) — Official experiment specification and question sheet.

## Conclusion

Experiment 2 highlighted the critical distinction between generative probabilistic models and lazy nearest-neighbor classifiers. For spam detection, Multinomial Naïve Bayes provides fast, high-precision filtering with minimal computational overhead. While hyperparameter-tuned KNN achieves higher peak accuracy (91.42%), its inference overhead and sensitivity to high-dimensional metric distortion make Naïve Bayes the preferred approach for high-throughput, low-latency spam classification.
