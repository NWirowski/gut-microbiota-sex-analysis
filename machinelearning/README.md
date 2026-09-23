# Machine Learning Analysis

This folder contains the Python scripts used for the revised machine learning analyses comparing participants with **Mood Disorder** (Bipolar Disorder or Major Depressive Disorder) with **Healthy Controls**.

The revised diagnostic classification combines Bipolar Disorder (BD) and Major Depressive Disorder (MDD) into a single **Mood Disorder** group. The original machine learning analyses based on the three-group classification (BD, MDD, and Control) are retained separately in [`/original_group_analyses`](../original_group_analyses/).

## Study Groups

The outcome variable is defined as:

| Group             | Coding                                              |
| ----------------- | --------------------------------------------------- |
| **Control**       | 0                                                   |
| **Mood Disorder** | 2 = Bipolar Disorder; 3 = Major Depressive Disorder |

For machine learning, BD and MDD are combined into a single binary **Mood Disorder** class.

## Predictor Sets

Two predictor sets are evaluated.

### Combined model

The **combined model** uses:

* 2,496 microbiome ASVs
* Sex

Thus, the feature matrix contains the complete microbiome profile together with sex as an additional predictor.

### Microbiome-only model

The **microbiome-only model** uses:

* 2,496 microbiome ASVs

Sex is not included as a predictor.

The microbiome-only analyses are implemented in scripts with the `_nosex` suffix.

| Script type      | Predictors       |
| ---------------- | ---------------- |
| Standard scripts | 2,496 ASVs + sex |
| `_nosex` scripts | 2,496 ASVs only  |

The two predictor sets allow assessment of model performance with and without sex included as an additional feature.

## Data Preparation

The machine learning analyses use the `table.csv` file.

Before model fitting:

1. The original diagnostic variable is recoded into Control and Mood Disorder.
2. The participant identifier (`id`) is removed.
3. Sex is excluded from the microbiome-only feature matrix.
4. ASV counts are converted to **relative abundance** by row normalization.
5. Sex is added as a binary predictor only in the combined models.

The resulting microbiome feature matrix contains approximately **2,496 ASVs**.

## Machine Learning Algorithms

Thirteen supervised machine learning algorithms are evaluated:

* Logistic Regression
* Support Vector Classifier (SVC)
* Multilayer Perceptron (MLP)
* HistGradientBoosting
* Gaussian Naive Bayes
* Decision Tree
* Bagging
* AdaBoost
* K-Nearest Neighbors (KNN)
* CatBoost
* Random Forest
* Extra Trees
* Gradient Boosting

The models use predefined hyperparameters that are kept consistent across validation strategies.

## Preprocessing Configurations

Each algorithm is evaluated across eight preprocessing configurations based on three preprocessing components:

* **SMOTETomek** — class-imbalance correction
* **PCA** — linear dimensionality reduction
* **UMAP** — nonlinear dimensionality reduction

All eight combinations are evaluated:

| SMOTETomek | PCA | UMAP |
| ---------- | --- | ---- |
| No         | No  | No   |
| No         | No  | Yes  |
| Yes        | No  | No   |
| Yes        | No  | Yes  |
| No         | Yes | No   |
| No         | Yes | Yes  |
| Yes        | Yes | No   |
| Yes        | Yes | Yes  |

Preprocessing transformations are fitted using the training data within each validation split and then applied to the corresponding test data.

## Validation Strategies

Four validation strategies are implemented.

### Leave-One-Out Cross-Validation

Scripts:

* `ml_r1_th_loo.py`
* `ml_r1_th_loo_nosex.py`

Leave-One-Out Cross-Validation (LOO) is performed across the complete sample.

Each participant is left out once as the test observation, while the remaining participants are used for training.

Performance is calculated from the aggregated individual LOO predictions.

For uncertainty estimation, **2,000 bootstrap resamples** of the aggregated LOO predictions are generated with replacement. **95% percentile confidence intervals** are calculated for:

* Accuracy
* Balanced accuracy
* Recall
* Precision
* F1-score
* ROC-AUC

### Stratified K-Fold Cross-Validation

Scripts:

* `ml_r1_th_skf.py`
* `ml_r1_th_skf_nosex.py`

A **10-fold Stratified K-Fold Cross-Validation** is performed.

The class distribution is preserved across folds.

Performance is calculated for each fold and summarized using:

* Mean
* Standard deviation (SD)

The following metrics are reported:

* Accuracy
* Balanced accuracy
* Recall
* Precision
* F1-score
* ROC-AUC

### Repeated Stratified K-Fold Cross-Validation

Scripts:

* `ml_r1_th_rskf.py`
* `ml_r1_th_rskf_nosex.py`

Repeated Stratified K-Fold Cross-Validation is performed using:

* 10 folds
* 3 repetitions
* 30 fold evaluations in total

Performance is calculated for each fold and repetition and summarized using the **mean and standard deviation**.

The following metrics are reported:

* Accuracy
* Balanced accuracy
* Recall
* Precision
* F1-score
* ROC-AUC

### Train-Test Split

Scripts:

* `ml_r1_th_tts.py`
* `ml_r1_th_tts_nosex.py`

A fixed **80% training / 20% test split** is used with `random_state = 42`.

The training set is used for model fitting and preprocessing, while the test set is retained for performance evaluation.

To quantify uncertainty in the test-set performance estimates, **2,000 repeated subsamples containing 90% of the test observations** are generated without replacement.

The resulting **95% percentile confidence intervals** are reported for:

* Accuracy
* Balanced accuracy
* Recall
* Precision
* F1-score
* ROC-AUC

## Performance Metrics

The machine learning analyses report:

* **Accuracy**
* **Balanced accuracy**
* **Recall**
* **Precision**
* **F1-score**
* **ROC-AUC**

The **F1-score** is used as the primary metric for comparing model performance because it jointly considers precision and recall.

## Script Overview

| Script                    | Predictor set             | Validation / Analysis                |
| ------------------------- | ------------------------- | ------------------------------------ |
| `ml_r1_th_loo.py`         | 2,496 ASVs + sex          | Leave-One-Out CV                     |
| `ml_r1_th_loo_nosex.py`   | 2,496 ASVs                | Leave-One-Out CV                     |
| `ml_r1_th_skf.py`         | 2,496 ASVs + sex          | 10-fold Stratified CV                |
| `ml_r1_th_skf_nosex.py`   | 2,496 ASVs                | 10-fold Stratified CV                |
| `ml_r1_th_rskf.py`        | 2,496 ASVs + sex          | 10-fold × 3-repetition Stratified CV |
| `ml_r1_th_rskf_nosex.py`  | 2,496 ASVs                | 10-fold × 3-repetition Stratified CV |
| `ml_r1_th_tts.py`         | 2,496 ASVs + sex          | 80/20 Train-Test Split               |
| `ml_r1_th_tts_nosex.py`   | 2,496 ASVs                | 80/20 Train-Test Split               |
| `r1_curva_roc_revista.py` | Selected ASV models ± sex | LOO ROC analysis                     |

## ROC Curve Analysis

### `r1_curva_roc_revista.py`

This script performs a focused ROC analysis using **Leave-One-Out Cross-Validation** for selected model and preprocessing configurations.

The analysis evaluates both:

* Models with sex included
* Microbiome-only models without sex

The selected configurations include:

* Logistic Regression + SMOTETomek + PCA
* Logistic Regression + SMOTETomek
* SVC + SMOTETomek
* SVC + SMOTETomek + UMAP
* SVC + SMOTETomek + PCA

For each selected configuration, the script:

1. Performs LOO cross-validation.
2. Stores the individual participant-level predictions.
3. Calculates aggregated performance metrics.
4. Calculates ROC-AUC.
5. Generates the ROC curve from the aggregated LOO predictions.
6. Calculates 95% bootstrap confidence intervals using 2,000 bootstrap resamples.
7. Saves the complete analysis to a checkpoint file.

The analysis is performed both with and without sex, resulting in **10 model configurations** in total.

The script generates:

* `LOO_ROC_selected_results.pkl` — checkpoint containing complete results and ROC information
* `df_results_LOO_ROC_selected.csv` — summary of selected model results
* `Figure_ROC_LOO_selected.png` — ROC curve figure with panels for models with and without sex

The ROC figure uses black-and-white line styles to distinguish the selected model configurations.

## Outputs

The validation scripts generate CSV files containing model performance results.

Typical output files include:

* `df_results_LOO.csv`
* `df_results_StratifiedKFold.csv`
* `df_results_RSKF_mean_SD.csv`
* `df_results.csv`
* `df_results_LOO_ROC_selected.csv`

Depending on the validation strategy, the result tables include:

* Validation strategy
* Model
* Preprocessing configuration
* Performance metrics
* Confidence intervals or standard deviations
* Number of folds/repetitions or test observations
* Bootstrap/subsampling parameters where applicable

## Reproducibility

The analyses use fixed random seeds where applicable, including:

* `random_state = 42` for model fitting and validation procedures
* `random_state = 123` for bootstrap and test-set subsampling procedures

The scripts were developed for execution in a Python/Google Colab environment.

Required packages include:

* `pandas`
* `numpy`
* `scikit-learn`
* `imbalanced-learn`
* `umap-learn`
* `catboost`
* `tqdm`
* `matplotlib`
* `plotly`

The input file `table.csv` must be available in the working directory.

## Analysis Overview

| Component                    | Specification                    |
| ---------------------------- | -------------------------------- |
| Outcome                      | Mood Disorder vs Control         |
| Sample                       | 93 participants                  |
| Microbiome predictors        | 2,496 ASVs                       |
| Combined predictors          | 2,496 ASVs + sex                 |
| Microbiome-only predictors   | 2,496 ASVs                       |
| Machine learning algorithms  | 13                               |
| Preprocessing configurations | 8                                |
| Validation strategies        | LOO, SKF, RSKF, TTS              |
| LOO uncertainty              | 2,000 bootstrap resamples        |
| SKF summary                  | Mean ± SD                        |
| RSKF                         | 10 folds × 3 repetitions         |
| TTS                          | 80% training / 20% test          |
| TTS uncertainty              | 2,000 × 90% test-set subsampling |
| Confidence interval          | 95% percentile CI                |
| Primary performance metric   | F1-score                         |

## Relation to the Original Analyses

The machine learning scripts in this folder correspond to the **revised Mood Disorder vs Control analysis**.

The previous machine learning analyses based on separate Bipolar Disorder, Major Depressive Disorder, and Control groups are retained in:

[`/original_group_analyses/machinelearning`](../original_group_analyses/machinelearning/)

The revised analyses were developed to address the limitations associated with the small diagnostic subgroup sizes and to provide more robust performance estimates through multiple validation strategies and uncertainty estimation.
