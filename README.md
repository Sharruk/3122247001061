# ICS1512 – Machine Learning Algorithms Laboratory

This repository contains the complete laboratory curriculum, experimental implementations, and comprehensive research reports for the **ICS1512 – Machine Learning Algorithms Laboratory** course at **Sri Sivasubramaniya Nadar College of Engineering (SSN), Chennai**.

The laboratory covers foundational to advanced machine learning paradigms, including exploratory data analysis, probabilistic modeling, penalized linear regression, kernel methods, tree ensembles, boosting, stacking, dimensionality reduction with PCA, unsupervised clustering, and neural representations (Perceptrons and Multilayer Perceptrons).

---

## Student Information

- **Name:** Sharruk S
- **Register Number:** 3122247001061
- **Degree:** M.Tech (Integrated) Computer Science & Engineering
- **Semester:** V
- **Academic Year:** 2026–2027 (Odd)
- **Faculty:** Dr. Poreddy Ajay Kumar Reddy
- **Institution:** Sri Sivasubramaniya Nadar College of Engineering, Chennai

---

## Repository Structure

```text
3122247001061/
├── assign1/                                  # Experiment 1: Python ML Stack & Modular EDA
│   ├── ex1.ipynb
│   ├── ex1.html
│   ├── Sharruk_S_3122247001061_Assignment1.pdf
│   ├── question/
│   └── README.md
├── assign2/                                  # Experiment 2: Naïve Bayes & KNN Spam Filtering
│   ├── ex2.ipynb
│   ├── ex2.html
│   ├── ICS1512_Exp2_SpamHam_Sharruk_S_3122247001061.pdf
│   ├── images/
│   ├── question/
│   └── README.md
├── assign3/                                  # Experiment 3: Linear & Regularized Regression
│   ├── ex3.ipynb
│   ├── Experiment_3_Report_Sharruk_S_3122247001061.pdf
│   ├── question/
│   └── README.md
├── assign4/                                  # Experiment 4: Linear & Kernel Binary Classification
│   ├── ex4.ipynb
│   ├── Experiment_4_Report_Sharruk_S_3122247001061.pdf
│   ├── question/
│   └── README.md
├── assign5/                                  # Experiment 5: Decision Tree & Random Forest
│   ├── ex5.ipynb
│   ├── Experiment_5_Report_Sharruk_S_3122247001061.pdf
│   ├── question/
│   └── README.md
├── assign6/                                  # Experiment 6: Bagging, Boosting & Stacking
│   ├── ex6.ipynb
│   ├── Experiment_6_Report_Sharruk_S_3122247001061.pdf
│   ├── question/
│   └── README.md
├── assign7/                                  # Experiment 7: Dimensionality Reduction (PCA)
│   ├── ex7.ipynb
│   ├── Experiment_7_Report_Sharruk_S_3122247001061.pdf
│   ├── question/
│   └── README.md
├── assign8/                                  # Experiment 8: K-Means, DBSCAN & HAC Clustering
│   ├── ex8.ipynb
│   ├── Experiment_8_Report_Sharruk_S_3122247001061.pdf
│   ├── figures/                              # 24 Generated publication-quality figures (EPS & PNG)
│   ├── tables/                               # 15 Exported result tables (CSV & JSON)
│   ├── question/
│   └── README.md
├── assign9/                                  # Experiment 9: Perceptron vs Multilayer Perceptron
│   ├── ex9.ipynb
│   ├── Experiment_9_Report_Sharruk_S_3122247001061.pdf
│   ├── question/
│   └── README.md
├── datasets/                                 # Central repository dataset storage
├── utils.py                                  # Reusable exploratory data analysis utilities
├── Manual.tex                                # LaTeX laboratory manual template
└── README.md                                 # Root repository documentation
```

---

## Experiments

### Experiment 1 — Working with Python Packages: NumPy, SciPy, Scikit-Learn, Matplotlib
Explores the core scientific Python computing stack for machine learning workflows. Implements modular, reusable Exploratory Data Analysis routines for tabular and image datasets, profiling statistical distributions, missing values, skewness, and class balance across benchmark datasets (Loan Prediction, Pima Diabetes, MNIST).

**Topics:**
- Multi-dimensional array manipulation and vectorization with NumPy
- Tabular data profiling, missingness diagnostics, and summary statistics with Pandas
- Statistical distributions and correlation analysis with SciPy and Seaborn
- Grayscale image matrix processing, pixel distributions, and centroid image visualization
- Systematic mapping of real-world datasets to supervised/unsupervised tasks and feature selection techniques

[View Experiment 1 →](./assign1/)

---

### Experiment 2 — Email Spam/Ham Classification using Naïve Bayes and KNN
Conducts an empirical comparison between probabilistic models (Gaussian, Multinomial, Bernoulli Naïve Bayes) and instance-based learning (K-Nearest Neighbors) on the 57-feature Spambase dataset. Benchmarks spatial indexing algorithms (KDTree, BallTree, Brute-Force) in high dimensions and evaluates hyperparameter tuning via GridSearchCV versus RandomizedSearchCV.

**Topics:**
- Bayes' theorem and conditional independence assumptions
- Gaussian, Multinomial, and Bernoulli likelihood models
- K-Nearest Neighbors neighborhood dynamics ($k=1 \dots 10$) and distance weighting
- Spatial indexing complexity: KDTree and BallTree vs. Brute-force under the curse of dimensionality
- Hyperparameter optimization with GridSearchCV and RandomizedSearchCV
- 5-Fold cross-validation stability and latency analysis

[View Experiment 2 →](./assign2/)

---

### Experiment 3 — Regression Analysis using Linear and Regularized Models
Investigates continuous financial loan prediction using Ordinary Least Squares (OLS) Linear Regression and penalized shrinkage models (Ridge, Lasso, Elastic Net). Incorporates an end-to-end ColumnTransformer pipeline with 5-fold cross-validated hyperparameter optimization, residual diagnostics, learning curve tracking, and coefficient sparsity analysis.

**Topics:**
- Multiple Linear Regression and Ordinary Least Squares (OLS)
- Ridge Regression ($L_2$ Tikhonov regularization)
- Lasso Regression ($L_1$ penalty inducing sparse feature selection)
- Elastic Net Regression ($L_1 + L_2$ convex combination)
- Scikit-Learn preprocessing pipelines (median/mode imputation, scaling, one-hot encoding)
- 5-Fold GridSearchCV scoring on $R^2$, learning curve diagnostics, and generalization gap evaluation

[View Experiment 3 →](./assign3/)

---

### Experiment 4 — Binary Classification using Linear and Kernel-Based Models
Explores linear and non-linear margin classification on the Spambase dataset using Logistic Regression and Support Vector Machines (SVM). Compares Linear, Polynomial, Radial Basis Function (RBF), and Sigmoid kernels, evaluates soft-margin slack tradeoffs ($C$), and analyzes the bias-variance tradeoff across decision boundary geometries.

**Topics:**
- Logistic Regression log-odds formulation and cross-entropy minimization
- Support Vector Classifiers (SVC) and maximal margin hyperplanes
- The Kernel Trick: Linear, Polynomial ($d=3$), RBF, and Sigmoid kernel transformations
- Feature standardization via `StandardScaler` for geometric margin invariance
- 5-Fold GridSearchCV hyperparameter tuning ($C, \gamma, \text{degree}$)
- Generalization gap analysis and computational training efficiency

[View Experiment 4 →](./assign4/)

---

### Experiment 5 — Decision Tree and Random Forest: A Comparative Classification Study
Presents a comparative oncology classification study on the Wisconsin Diagnostic Breast Cancer (WDBC) dataset using individual Decision Trees and Random Forest ensembles. Analyzes splitting criteria (Entropy/Information Gain vs. Gini Impurity), pre-pruning tree regularization, bootstrap aggregation (bagging), random feature subspace projection, and Gini feature importances.

**Topics:**
- Recursive binary partitioning: Information Gain (Entropy) vs. Gini Impurity
- Tree pre-pruning and depth constraints (`max_depth`, `min_samples_leaf`, `min_samples_split`)
- Random Forest ensemble mechanics: Bootstrap aggregation and random subspace sampling
- 5-Fold cross-validation hyperparameter optimization
- Generalization variance reduction and ROC-AUC discrimination comparison
- Gini feature importance extraction for clinical biomarker discovery

[View Experiment 5 →](./assign5/)

---

### Experiment 6 — Bagging, Boosting, and Stacked Ensemble Models
Investigates three foundational ensembling paradigms—Bagging, Boosting (AdaBoost and Gradient Boosting), and Heterogeneous Stacking—applied to the Wisconsin Diagnostic Breast Cancer dataset. Evaluates 5-fold cross-validation stability (mean and standard deviation), probability calibration, and the distinct error-reduction mechanisms of each ensemble family.

**Topics:**
- Parallel variance reduction via `BaggingClassifier` with decision trees
- Sequential bias reduction via `AdaBoostClassifier` (exponential loss instance re-weighting)
- Gradient Tree Boosting (`GradientBoostingClassifier`) optimizing cross-entropy pseudo-residuals
- Heterogeneous Stacking (`StackingClassifier` with SVM, Naïve Bayes, and Decision Tree $\to$ Logistic Regression)
- 5-Fold cross-validation stability profiling ($\mu \pm \sigma$ across folds)
- Bias-variance decomposition analysis and ROC-AUC calibration

[View Experiment 6 →](./assign6/)

---

### Experiment 7 — Dimensionality Reduction and Model Evaluation (With and Without PCA)
Studies the effect of unsupervised linear dimensionality reduction via Principal Component Analysis (PCA) by compressing 30 continuous features down to 10 orthogonal principal components (preserving 95.21% cumulative variance). Evaluates 10 distinct classifiers across both No-PCA and With-PCA conditions to analyze inductive bias interactions with orthogonal feature rotation.

**Topics:**
- Principal Component Analysis: Covariance matrix eigen-decomposition and scree spectrum analysis
- Cumulative variance retention ($\ge 95\%$) and feature space compression (3:1 ratio)
- Paired evaluation across 10 classifiers (SVM, Naïve Bayes, KNN, Logistic Regression, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, XGBoost, Stacking)
- Systematic hyperparameter tuning within reduced eigenspaces
- Inductive bias analysis: why margin-based models improve while axis-aligned tree splits degrade under PCA

[View Experiment 7 →](./assign7/)

---

### Experiment 8 — Clustering Human Activity Recognition Data using K-Means, DBSCAN, and Hierarchical Clustering
Applies unsupervised clustering algorithms to high-dimensional accelerometer and gyroscope time-series data (561 features across 10,299 instances) from the UCI Human Activity Recognition dataset. Benchmarks K-Means, DBSCAN, and Hierarchical Agglomerative Clustering (Ward, Complete, Average, Single linkage), evaluating internal cluster cohesion/separation, external ground-truth agreement (ARI, NMI), and 2D projections via PCA and t-SNE.

**Topics:**
- Partition-based clustering: K-Means optimization with Elbow (WCSS) and Silhouette validation
- Density-based spatial clustering: DBSCAN parameter tuning ($\epsilon, \text{min\_samples}$) and high-dimensional noise isolation
- Hierarchical Agglomerative Clustering (HAC) and dendrogram construction
- Linkage evaluation (Ward's minimum variance, Complete, Average, Single) and Cophenetic correlation
- Internal cluster validation: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index
- External validation against 6 ground-truth activities: Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), and contingency confusion matrices
- Manifold visualization: PCA variance decomposition (revealing the Dynamic vs. Static macro-split) and 2D t-SNE projections

[View Experiment 8 →](./assign8/)

---

### Experiment 9 — Perceptron vs Multilayer Perceptron (A/B Experiment) with Hyperparameter Tuning
Presents a controlled A/B empirical study on the 62-class English Handwritten Characters (Chars74K "EnglishHnd") dataset. Benchmarks a from-scratch Single-Layer Perceptron Learning Algorithm (PLA) implemented in pure NumPy via One-vs-Rest (OvR) against a backpropagation-trained Multilayer Perceptron (MLP), demonstrating the theoretical and practical necessity of non-linear hidden representations for complex image recognition.

**Topics:**
- Single-Layer Perceptron Learning Algorithm (PLA) implemented from scratch in pure NumPy
- One-vs-Rest (OvR) multiclass extension for 62 alphanumeric character classes
- Multilayer Perceptrons (MLP) and backpropagation with cross-entropy loss
- High-resolution scan preprocessing: Grayscale conversion, $32 \times 32$ bilinear interpolation, intensity normalization, and vector flattening (1,024 features)
- Systematic hyperparameter tuning: Activation functions (`logistic`, `relu`, `tanh`), optimizers (`adam`, `sgd`), learning rates, batch sizes, and hidden layer topologies
- Strict 70/15/15 train/validation/test evaluation protocol
- Representation learning, multiclass confusion matrix diagnostics, and ROC-AUC evaluation

[View Experiment 9 →](./assign9/)

---

## Tools & Libraries

The implementations throughout this laboratory rely on the standard Python scientific computing and machine learning stack:

- **Core Programming:** Python 3.12 / 3.13
- **Numerical & Matrix Computing:** NumPy
- **Data Manipulation & Wrangling:** Pandas
- **Scientific & Statistical Computing:** SciPy
- **Classical Machine Learning:** Scikit-Learn
- **Gradient Boosting Frameworks:** XGBoost
- **Data Visualization:** Matplotlib, Seaborn
- **Image Processing:** Pillow (PIL)
- **Interactive Development:** Jupyter Notebook

---

## Datasets

The repository includes a centralized `datasets/` directory containing all public benchmarks explored across the laboratory curriculum:

- **`breastCancer/`** — Wisconsin Diagnostic Breast Cancer (WDBC) dataset (569 instances, 30 cellular morphology features, binary diagnosis). Used in Experiments 5, 6, and 7.
- **`diabetes/`** — Pima Indians Diabetes dataset (768 instances, 8 clinical predictors, binary diabetes outcome). Used in Experiment 1.
- **`loan_prediction/`** — Loan Prediction Problem dataset (614 instances, financial and demographic attributes, continuous loan amount target). Used in Experiments 1 and 3.
- **`spambase.csv`** — Spambase email classification dataset (4,601 instances, 57 word/character frequency attributes, binary spam indicator). Used in Experiments 2 and 4.
- **`mnist/`** — MNIST handwritten digit dataset ($28 \times 28$ grayscale pixel arrays, 10 digit classes). Used in Experiment 1.
- **`UCI HAR Dataset/`** — Human Activity Recognition Using Smartphones dataset (10,299 instances, 561 time/frequency features, 6 physical activity classes). Used in Experiment 8.
- **`English/`** — English Handwritten Characters Dataset (Chars74K "EnglishHnd" subset, 3,410 images across 62 alphanumeric classes). Used in Experiment 9.

---

## Repository Contents

- **`assign1/` to `assign9/`** — Individual experiment directories containing full source code notebooks (`.ipynb`), HTML exports, official laboratory reports (`.pdf`), question sheets, and detailed documentation.
- **`datasets/`** — Raw and formatted datasets utilized across the assignments.
- **`utils.py`** — Reusable helper module containing automated tabular (`eda_tabular`) and image (`eda_images`) exploratory data analysis routines.
- **`Manual.tex`** — Institutional LaTeX laboratory manual template defining report standards, figures, tables, and inference structures.

---

## Author

**Sharruk S**  
Register Number: **3122247001061**  
M.Tech (Integrated) Computer Science & Engineering, Semester V  
Sri Sivasubramaniya Nadar College of Engineering, Chennai  
Faculty: Dr. Poreddy Ajay Kumar Reddy