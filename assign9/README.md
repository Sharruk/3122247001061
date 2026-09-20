# Experiment 9 — Perceptron vs Multilayer Perceptron (A/B Experiment) with Hyperparameter Tuning

## Overview

This experiment presents a controlled A/B comparative evaluation between a single-layer **Perceptron Learning Algorithm (PLA)** and a **Multilayer Perceptron (MLP)** on the challenging 62-class **English Handwritten Characters (Chars74K "EnglishHnd")** dataset. The study explores the fundamental limitation of linear decision boundaries on non-linearly separable image glyphs. A multiclass PLA is implemented **from scratch** using pure NumPy via a One-vs-Rest (OvR) scheme and benchmarked against an MLP equipped with non-linear hidden representations trained via backpropagation with systematic hyperparameter tuning.

## Objectives

- Implement the classical Single-Layer **Perceptron Learning Algorithm (PLA) from scratch** using only NumPy, extending it to a 62-class classification problem via a **One-vs-Rest (OvR)** architecture.
- Construct a **Multilayer Perceptron (MLP)** with non-linear activations and backpropagation.
- Preprocess real-world handwriting scans (converting high-res RGB images to normalized $32 \times 32$ grayscale feature vectors).
- Perform systematic hyperparameter optimization for the MLP across:
  - Activation functions (`logistic`, `relu`, `tanh`)
  - Optimization algorithms (`adam`, `sgd`)
  - Learning rates ($\eta \in \{0.0005, 0.001, 0.005\}$)
  - Mini-batch sizes ($B \in \{16, 32, 64\}$)
  - Hidden layer architectures (single- and multi-layer topologies)
- Execute a strict **A/B test** on a held-out test split (15%, 512 images), comparing Accuracy, Macro/Weighted Precision, Recall, $F_1$-score, Micro/Macro ROC-AUC, and training latency.
- Analyze epoch-wise convergence dynamics, multiclass confusion patterns, and the mathematical roots of PLA's failure on non-linear image manifolds.

## Concepts Covered

- **Perceptron Learning Algorithm (PLA)**:
  - Threshold activation: $f(z) = +1$ if $z \ge 0$, else $-1$.
  - Perceptron weight update rule for misclassified instances:
    $$w_{t+1} = w_t + \eta (y_i - \hat{y}_i) x_i$$
  - Novikoff's Perceptron Convergence Theorem and its breakdown when data is linearly non-separable.
- **One-vs-Rest (OvR) Multiclass Reduction**: Training $C=62$ independent binary perceptrons and taking $\hat{c} = \arg\max_{c} (w_c^T x + b_c)$.
- **Multilayer Perceptrons & Backpropagation**:
  - Non-linear activations ($\text{ReLU}(z) = \max(0, z)$, $\text{Logistic}(z) = \frac{1}{1 + e^{-z}}$, $\tanh(z)$).
  - Cross-entropy loss for multi-class classification:
    $$\mathcal{L} = -\sum_{c=1}^C y_c \log \hat{p}_c$$
  - Generalized delta rule and gradient descent through hidden layers.
- Representation learning: transforming raw pixels into linearly separable latent feature representations.

## Algorithms / Techniques

- **Model A**: `MulticlassPerceptronOvR` (implemented from scratch in pure NumPy, 62 parallel perceptrons)
- **Model B**: `MLPClassifier` (Scikit-learn neural network module)
- **Image Pipeline**: Resizing to $32 \times 32$, grayscale conversion (`Pillow`), intensity normalization to $[0, 1]$, and flattening to 1,024 features
- **Hyperparameter Exploration**: Systematic validation grid over activations, solvers, learning rates, batch sizes, and hidden layers
- **Validation Protocol**: Strict 70% Train (2,386 images), 15% Validation (512 images), and 15% Test (512 images) stratified split

## Dataset

- **Dataset Name:** English Handwritten Characters Dataset (Chars74K "EnglishHnd" subset)
- **Location:** `datasets/English/Hnd/Img/`
- **Total Samples:** 3,410 images
- **Number of Classes:** 62 alphanumeric character classes:
  - `Sample001` – `Sample010`: Digits `0` through `9`
  - `Sample011` – `Sample036`: Uppercase letters `A` through `Z`
  - `Sample037` – `Sample062`: Lowercase letters `a` through `z`
- **Samples per Class:** Exactly 55 instances per class (produced by 55 distinct human writers)
- **Preprocessing Dimensions:**
  - Raw Scans: Variable high-resolution RGB (~$1200 \times 900$ pixels)
  - Processed Inputs: $32 \times 32$ single-channel grayscale arrays $\to$ 1,024-dimensional flattened feature vectors in $[0, 1]$

## Implementation

The notebook ([ex9.ipynb](./ex9.ipynb)) implements the complete experimental pipeline:

1. **Data Ingestion & Verification:** Discovers and validates all 3,410 image files across 62 directory folders without hardcoding file counts.
2. **Preprocessing Pipeline:** Converts images to grayscale, resizes them using bilinear interpolation to $32 \times 32$, normalizes pixel values to $[0, 1]$, and verifies class balance.
3. **Dataset Partitioning:** Splits data into 70% training (2,386), 15% validation (512), and 15% testing (512) using stratified sampling.
4. **Model A (PLA from Scratch):**
   - Implements 62 independent binary perceptrons using vectorized NumPy operations.
   - Trains for 30 epochs with learning rate $\eta = 0.1$.
   - Tracks per-epoch binary error rate and multiclass training accuracy.
5. **Model B (MLP Hyperparameter Tuning):**
   - Explores multiple architectures (`(64,)`, `(128,)`, `(256,)`, `(128, 64)`, `(256, 128)`).
   - Tests activations (`logistic`, `relu`, `tanh`), optimizers (`adam`, `sgd`), learning rates ($0.0005, 0.001, 0.005$), and batch sizes ($16, 32, 64$).
   - Evaluates each configuration strictly on the validation set to prevent test set contamination.
6. **Final Model Training & Evaluation:** Refits the optimal MLP configuration and evaluates both PLA and MLP on the held-out test set.
7. **Diagnostics & Visualization:** Generates sample glyph grids, class-average images, training convergence curves, 62-class confusion matrices, and multiclass ROC-AUC curves.

## Results / Analysis

### 1. MLP Hyperparameter Tuning Results (Validation Set)

| Config ID | Hidden Layer Architecture | Activation | Optimizer | Initial Learning Rate | Batch Size | Validation Accuracy (%) |
|---|---|---|---|---|---|---|
| **`C03` (Selected)** | **`(128,)`** | **`logistic`** | **`adam`** | **0.001** | **32** | **39.45%** |
| `C01` | `(128,)` | `relu` | `adam` | 0.001 | 32 | 36.33% |
| `C02` | `(128,)` | `tanh` | `adam` | 0.001 | 32 | 35.16% |
| `C04` | `(128,)` | `logistic` | `sgd` | 0.001 | 32 | 1.95% (stalled) |
| `C05` | `(256, 128)` | `logistic` | `adam` | 0.001 | 32 | 37.11% |
| `C06` | `(64,)` | `logistic` | `adam` | 0.001 | 32 | 33.20% |

*Inference:* Logistic (sigmoid) activation paired with the `adam` optimizer yielded the highest validation accuracy ($39.45\%$). Standard `sgd` at $\eta = 0.001$ failed to escape saddle points within 50 epochs due to vanishing gradients across 62 output nodes. Adding a second hidden layer (`(256, 128)`) did not improve accuracy due to the small sample size (only 38 training samples per class), causing mild parameter overfitting.

### 2. Final A/B Model Comparison (Held-Out Test Set: 512 Images)

| Metric | Model A: Single-Layer PLA (OvR from Scratch) | Model B: Tuned Multilayer Perceptron (MLP) | Absolute Improvement | Relative Improvement |
|---|---|---|---|---|
| **Test Accuracy** | **17.97%** (0.1797) | **38.87%** (0.3887) | **+20.90%** | **+116.3%** |
| **Macro Precision** | 0.2245 | 0.4249 | +0.2004 | +89.3% |
| **Macro Recall** | 0.1812 | 0.3884 | +0.2072 | +114.4% |
| **Macro $F_1$-Score** | **0.1425** | **0.3872** | **+0.2447** | **+171.7%** |
| **Weighted $F_1$-Score** | 0.1421 | 0.3882 | +0.2461 | +173.2% |
| **Micro ROC-AUC** | 0.7723 | **0.9128** | +0.1405 | +18.2% |
| **Macro ROC-AUC** | 0.8343 | **0.9090** | +0.0747 | +9.0% |
| **Training Epochs** | 30 epochs | 120 epochs | +90 epochs | — |
| **Training Time** | **9.01 s** | 12.39 s | +3.38 s | — |

### 3. Key Observations & Inferences

1. **Why does PLA underperform on handwritten characters?**
   - Single-layer PLA is strictly limited to linear decision boundaries ($w^T x + b = 0$).
   - Handwritten characters from 55 different human writers feature substantial stylistic variations—differing stroke slants, loops, curvatures, and line thicknesses.
   - Classes sharing stroke primitives (such as `0`, `O`, `o`, or `1`, `I`, `l`, or `S`, `5`) are mutually non-linearly separable in raw pixel space. PLA oscillates without converging, hitting a ceiling at $17.97\%$ test accuracy.
2. **How does the MLP resolve non-linear separability?**
   - The MLP's 128 hidden neurons act as learned non-linear basis functions, mapping the raw 1,024-dimensional pixel space into an internal manifold where stroke geometries (loops, horizontal bars, ascenders/descenders) become linearly separable.
   - This architectural capacity enables the MLP to achieve **38.87% accuracy** and **0.9128 micro ROC-AUC** on a 62-class problem where random guessing yields only $1.61\%$.
3. **Optimizer Impact (Adam vs. SGD):**
   - Adam's adaptive learning rates and momentum enabled rapid convergence across sparse gradient vectors, whereas vanilla SGD stalled at near-random accuracy ($1.95\%$) when unassisted by momentum and adaptive step sizing.
4. **Error Distribution:**
   - Multiclass confusion matrix analysis revealed that errors cluster predictably between casing pairs with identical glyph geometries (e.g., `C`/`c`, `O`/`o`, `S`/`s`, `X`/`x`, `Z`/`z`), where size is ambiguous without contextual baseline guides.

## Files

- [ex9.ipynb](./ex9.ipynb) — Complete Jupyter notebook implementing the PLA from scratch, MLP model architectures, tuning experiments, and evaluation plots.
- [Experiment_9_Report_Sharruk_S_3122247001061.pdf](./Experiment_9_Report_Sharruk_S_3122247001061.pdf) — Comprehensive laboratory report containing mathematical formulations, PLA update derivations, MLP backpropagation equations, tuning tables, confusion matrices, and detailed analytical answers.
- [question/Experiment_9.pdf](./question/Experiment_9.pdf) — Official experiment assignment brief.

## Conclusion

Experiment 9 provided an empirical demonstration of the perceptual limitations of single-layer linear networks and the necessity of deep, non-linear representations. On a 62-class handwriting recognition task with natural writer variability, the **Multilayer Perceptron outperformed the from-scratch Perceptron by more than 2:1 in accuracy (38.87% vs. 17.97%)** and **nearly 3:1 in macro $F_1$-score (0.3872 vs. 0.1425)**, while achieving an outstanding **0.9128 micro ROC-AUC**.
