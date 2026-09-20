# Experiment 8 — Clustering Human Activity Recognition Data using K-Means, DBSCAN, and Hierarchical Clustering

## Overview

This experiment explores unsupervised machine learning on high-dimensional sensor data from the **UCI Human Activity Recognition (HAR)** dataset. It implements, evaluates, and compares three fundamental clustering paradigms: **K-Means** (centroid partitioning), **DBSCAN** (density-based spatial clustering), and **Hierarchical Agglomerative Clustering (HAC)** (connectivity-based clustering with Ward, Complete, Average, and Single linkages). The analysis evaluates both internal cluster validation criteria (Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index) and external ground-truth agreement (Adjusted Rand Index, Normalized Mutual Information), with 2D manifold projections via **PCA** and **t-SNE**.

## Objectives

- Apply unsupervised clustering algorithms to high-dimensional time-and-frequency sensor data (561 features across 10,299 samples).
- Implement and optimize three distinct clustering models:
  - **Model A: K-Means Clustering** — Determine optimal $k$ via the **Elbow method** (WCSS / Inertia) and **Silhouette analysis** across $k \in [2, \dots, 8]$.
  - **Model B: DBSCAN** — Select neighborhood radius $\epsilon$ and $\text{min\_samples}$ using $k$-distance elbow plots and systematic parameter grid search.
  - **Model C: Hierarchical Agglomerative Clustering (HAC)** — Benchmark **Ward**, **Complete**, **Average**, and **Single** linkage criteria using Cophenetic correlation coefficients and dendrogram visualizations.
- Evaluate clustering quality using **internal validation metrics**:
  - Silhouette Score (cohesion vs. separation)
  - Davies-Bouldin Index (ratio of within-cluster scatter to between-cluster separation)
  - Calinski-Harabasz Index (variance ratio criterion)
- Benchmark discovered clusters against the 6 physical activity ground-truth labels using **external validation metrics**:
  - Adjusted Rand Index (ARI)
  - Normalized Mutual Information (NMI)
  - Contingency matrices (cluster-to-activity confusion mapping)
- Uncover the latent geometric structure of human activity data using **Principal Component Analysis (PCA)** and **t-Distributed Stochastic Neighbor Embedding (t-SNE)**.

## Concepts Covered

- **K-Means Objective**:
  $$J = \sum_{j=1}^k \sum_{x_i \in C_j} \|x_i - \mu_j\|^2$$
  Iterative Expectation-Maximization minimizing within-cluster sum of squares (WCSS).
- **DBSCAN Density Formulations**:
  - $\epsilon$-neighborhood: $N_\epsilon(p) = \{q \in D \mid \text{dist}(p, q) \le \epsilon\}$
  - Core points ($|N_\epsilon(p)| \ge \text{min\_samples}$), Border points, and Noise points.
  - Robustness to arbitrary cluster geometries and automated outlier/noise rejection.
- **Hierarchical Agglomerative Clustering (HAC)**:
  - Bottom-up greedy merging based on inter-cluster distance metrics.
  - Ward's minimum variance criterion: $\Delta \text{ESS}_{AB} = \frac{n_A n_B}{n_A + n_B} \|\mu_A - \mu_B\|^2$
  - Cophenetic Correlation Coefficient measuring how faithfully a dendrogram preserves pairwise input distances.
- **Curse of Dimensionality in Clustering**: Distance concentration phenomena in 561-dimensional Euclidean space and its severe impact on density estimation.
- Internal validation (unsupervised) vs. External validation (supervised ground-truth comparison).

## Algorithms / Techniques

- **Clustering Algorithms**: `KMeans`, `DBSCAN`, `AgglomerativeClustering`
- **Dimensionality Reduction**: `PCA` (variance scree and 2D projections) and `TSNE` (non-linear manifold visualization)
- **Hierarchical Tooling**: `scipy.cluster.hierarchy` (`linkage`, `dendrogram`, `cophenet`, `fcluster`)
- **Spatial Geometry**: `NearestNeighbors` for $k$-distance percentile estimation and chord-distance elbow detection
- **Metrics**: `silhouette_score`, `davies_bouldin_score`, `calinski_harabasz_score`, `adjusted_rand_score`, `normalized_mutual_info_score`

## Dataset

- **Dataset Name:** Human Activity Recognition Using Smartphones Dataset (`datasets/UCI HAR Dataset`)
- **Source:** UCI Machine Learning Repository
- **Number of Samples:** 10,299 instances (Train: 7,352 + Test: 2,947 pooled for unsupervised discovery)
- **Number of Features:** 561 continuous variables extracted from 3-axis linear acceleration and 3-axis angular velocity signals captured at 50 Hz
- **Subjects:** 30 volunteers aged 19–48 years
- **Ground-Truth Activities (6 Classes):**
  1. `WALKING` (1,722 samples)
  2. `WALKING_UPSTAIRS` (1,544 samples)
  3. `WALKING_DOWNSTAIRS` (1,406 samples)
  4. `SITTING` (1,777 samples)
  5. `STANDING` (1,906 samples)
  6. `LAYING` (1,944 samples)
- **Data Quality:** 0 missing values; 0 duplicate rows; features pre-bounded in $[-1, 1]$ and normalized using `StandardScaler`.

## Implementation

The implementation ([ex8.ipynb](./ex8.ipynb)) executes the following rigorous pipeline:

1. **Dataset Loading & Preprocessing:** Merges train and test splits into a unified 10,299-sample matrix, validates missingness/duplicates, and applies `StandardScaler`.
2. **Exploratory Data Analysis:** Profiles activity balance (ratio: 1.38), feature distributions, and correlation structure (21.4% of feature pairs exhibit $|\rho| > 0.7$).
3. **PCA Decomposition:** Analyzes the eigenvalue spectrum:
   - **PC1:** Accounts for **50.74%** of total variance.
   - **PC2:** Accounts for **6.24%** (Cumulative $\text{PC1} + \text{PC2} = \mathbf{56.98\%}$).
   - 90% variance captured by 65 PCs; 95% by 104 PCs; 99% by 182 PCs.
4. **Model A (K-Means):** Evaluates $k \in [2, \dots, 8]$ with 10 initializations each. Calculates WCSS reduction, silhouette profiles, and maximum perpendicular chord distance to identify the elbow.
5. **Model B (DBSCAN):** Evaluates a 44-parameter grid spanning $\epsilon \in [9, \dots, 20]$ and $\text{min\_samples} \in [5, 10, 20, 50]$ guided by a $k$-distance plot. Selects the optimal configuration based on internal validation criteria.
6. **Model C (Hierarchical Clustering):** Benchmarks Ward, Complete, Average, and Single linkage across $K=2 \dots 8$. Computes cophenetic correlations, generates truncated and full dendrograms, and checks partition stability.
7. **Cross-Model Comparison & Manifold Visualizations:** Evaluates all final models side-by-side using internal metrics, external ground-truth alignment, contingency heatmaps, and 2D PCA/t-SNE cluster maps.

## Results / Analysis

### 1. Final Clustering Model Comparison Table

| Model | Algorithm | Hyperparameters | Clusters | Noise Points | Silhouette Score | Davies-Bouldin | Calinski-Harabasz | ARI | NMI |
|---|---|---|---|---|---|---|---|---|---|
| **K-Means ($k=2$)** | K-Means | $k=2, n_{\text{init}}=10$ | 2 | 0 | 0.3937 | 1.0707 | **7880.81** | 0.3296 | 0.5455 |
| **K-Means ($k=6$)** | K-Means | $k=6, n_{\text{init}}=10$ | 6 | 0 | 0.1099 | 2.3836 | 2556.54 | 0.4196 | 0.5593 |
| **DBSCAN** | DBSCAN | $\epsilon=14, \text{min\_samples}=50$ | 2 | 4,126 (40.1%) | **0.4077** | **1.0572** | 3343.37 | 0.2574 | 0.4051 |
| **HAC-Ward ($K=2$)** | Hierarchical | $\text{linkage}=\text{ward}, K=2$ | 2 | 0 | 0.3934 | 1.0717 | 7859.76 | 0.3325 | 0.5568 |
| **HAC-Ward ($K=6$)** | Hierarchical | $\text{linkage}=\text{ward}, K=6$ | 6 | 0 | 0.1170 | 2.4820 | 2349.67 | **0.4599** | **0.6015** |

### 2. Hierarchical Linkage Cophenetic Correlation

| Linkage Criterion | Cophenetic Correlation Coefficient | Cluster Behavior & Topology |
|---|---|---|
| **Average Linkage** | **0.8557** | Best pairwise distance preservation; balanced tree merges. |
| **Single Linkage** | 0.8139 | Severe **chaining effect**; isolates outliers into singletons and leaves one giant cluster. |
| **Complete Linkage** | 0.7523 | Enforces compact spherical clusters; sensitive to outlier bridges. |
| **Ward's Method** | 0.6767 | **Best practical clusters**; minimizes within-cluster variance, producing the highest ARI/NMI. |

### 3. Key Discoveries & Physical Observations

1. **The Dynamic vs. Static Dichotomy:**
   - Principal Component 1 (accounting for **50.74%** of variance) cleanly separates all samples into two macroscopic physical regimes:
     - **Dynamic Activities** (`WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`) characterized by high acceleration variance.
     - **Static Postures** (`SITTING`, `STANDING`, `LAYING`) characterized by near-constant gravity acceleration vectors.
   - This physical reality is why all internal metrics (Silhouette, Davies-Bouldin) peak decisively at **$k=2$**.
2. **DBSCAN in 561-Dimensional Space:**
   - In 561 dimensions, pairwise Euclidean distances become tightly concentrated (mean distance: ~30.5).
   - DBSCAN requires an exceptionally large neighborhood radius ($\epsilon=14$) to discover core point connectivity, resulting in **4,126 points ($40.06\%$) labeled as noise**.
   - While the non-noise core points form tight clusters (yielding a high Silhouette score of $0.4077$), DBSCAN is unsuited for complete activity partitioning due to uniform density decay in high dimensions.
3. **Superiority of Ward's Hierarchical Clustering at $K=6$:**
   - When configured for the 6 actual activities ($K=6$), **HAC with Ward's linkage** achieved the highest external agreement across the entire experiment:
     - $\text{ARI} = \mathbf{0.4599}$ and $\text{NMI} = \mathbf{0.6015}$ (outperforming K-Means at $\text{ARI}=0.4196$, $\text{NMI}=0.5593$).
   - Contingency analysis shows that Ward cleanly separates the dynamic activities and successfully isolates `LAYING` from `SITTING`/`STANDING`.
4. **Internal vs. External Metric Tension:**
   - Internal metrics reward separation along the dominant PC1 variance axis, choosing $k=2$.
   - External metrics reward fine-grained discrimination among the 6 domain activities, selecting $k=6$.

## Files

- [ex8.ipynb](./ex8.ipynb) — Complete Jupyter notebook implementing K-Means, DBSCAN, and HAC with 78 detailed cells.
- [Experiment_8_Report_Sharruk_S_3122247001061.pdf](./Experiment_8_Report_Sharruk_S_3122247001061.pdf) — Comprehensive publication-quality laboratory report with full mathematical foundations, dendrograms, contingency tables, and discussion.
- [figures/](./figures/) — Directory holding **24 generated publication figures** (in both vector `.eps` and high-res `.png` formats):
  - `fig01_activity_distribution`: Class frequency bar chart
  - `fig05_pca_scree`: Scree plot and cumulative explained variance
  - `fig07_kmeans_elbow` & `fig08_kmeans_silhouette`: K-Means selection curves
  - `fig10_dbscan_kdistance` & `fig11_dbscan_tuning_heatmaps`: DBSCAN tuning diagnostics
  - `fig14_dendrogram_ward_truncated` & `fig15_dendrograms_all_linkages`: Hierarchical dendrograms
  - `fig21_side_by_side_pca` & `fig22_side_by_side_tsne`: Comprehensive 2D cluster comparison maps
  - `fig23_contingency_selected_k` & `fig24_contingency_k6_and_dbscan`: Cluster-vs-activity confusion matrices
- [tables/](./tables/) — Directory containing **15 structured result tables** (CSVs and JSON):
  - `results.json`: Machine-readable parameters, variance ratios, and evaluation scores
  - `table1_kmeans_elbow_silhouette.csv`: K-Means metrics across $k=2 \dots 8$
  - `table2_final_clustering_comparison.csv`: Head-to-head comparison of all 5 final models
  - `table3_dbscan_parameter_tuning.csv`: Full 44-configuration DBSCAN grid search results
  - `table4_hierarchical_linkage_comparison.csv`: Cophenetic and validation scores across linkages
  - `contingency_*.csv`: Detailed cluster-vs-activity mapping tables
- [question/Experiment_8.pdf](./question/Experiment_8.pdf) — Official experiment assignment brief.

## Conclusion

Experiment 8 provided deep insights into the behavior of unsupervised algorithms on complex, high-dimensional real-world data. Centroid-based (K-Means) and minimum-variance hierarchical (Ward) clustering successfully captured the underlying biomechanical structure of the dataset. While the data strongly clusters into two macro-states (Dynamic vs. Static) along the primary principal component, **Ward's Hierarchical Clustering at $K=6$** proved to be the most effective algorithm for recovering the six individual activities, achieving an Adjusted Rand Index of $0.4599$ and Normalized Mutual Information of $0.6015$.
