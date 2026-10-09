# Machine Learning Classification of Mood Disorders Using Gut Microbiome Data

## Overview

This repository contains machine learning analyses designed to classify individuals with mood disorders (TH) and healthy controls using gut microbiome data.

The project evaluates 13 machine learning algorithms under four validation strategies, comparing three predictor configurations: microbiome features combined with sex, microbiome features alone, and sex alone.

The classification task is consistent across all analyses: **mood disorder (TH) versus healthy controls**. The scripts do not implement separate classification tasks for bipolar disorder versus controls, depression versus controls, or bipolar disorder versus depression.

The analyses also investigate preprocessing configurations for microbiome-based models and generate performance tables and receiver operating characteristic (ROC) curve visualizations.

## Objectives

- Evaluate the performance of multiple machine learning algorithms in classifying mood disorder status.
- Compare models using microbiome features combined with sex, microbiome features alone, and sex alone.
- Assess model performance using four validation strategies.
- Investigate the effects of preprocessing techniques on microbiome-based classification.
- Generate performance tables and ROC curve visualizations for model comparison.

## Dataset and Target Variable

The scripts use `table.csv` as the input dataset. The file must be available in the working directory expected by the scripts.

The dataset contains microbiome amplicon sequence variant (ASV) features and participant information, including sex and mood disorder status. The microbiome analyses use 2,496 ASVs.

### Binary classification

All scripts address the same binary classification task:

- **Mood disorder (TH):** participants assigned to the mood disorder group.
- **Healthy controls:** participants assigned to the control group.

The original `mood` variable is recoded so that the mood disorder categories coded as `2` and `3` are combined into a single TH class, while the category coded as `0` represents healthy controls.

The resulting binary target is:

- `0` / `False`: Healthy control
- `1` / `True`: Mood disorder (TH)

The participant identifier (`id`) is excluded from the predictors.

## Predictor Configurations

Three predictor configurations are evaluated.

### 1. Microbiome + Sex

This configuration combines microbiome ASV features with sex as predictors.

It evaluates classification performance using both microbial information and sex.

### 2. Microbiome Only

This configuration uses microbiome ASV features without sex as a predictor.

These models allow performance to be assessed using microbiome information alone. The corresponding scripts are identified by the `_nosex` suffix.

### 3. Sex Only

This configuration uses sex as the sole predictor, without microbiome features.

The sex-only models provide a baseline against which the microbiome-based models can be contextualized. These analyses do not apply SMOTETomek, PCA, or UMAP.

The sex-only scripts encode sex using `df["sex"] == 1`. The interpretation of this encoding depends on the original coding of the `sex` variable in `table.csv`.

## Machine Learning Algorithms

The analyses evaluate the following 13 machine learning algorithms:

1. Logistic Regression
2. Support Vector Classifier (SVC)
3. Multi-Layer Perceptron (MLP)
4. HistGradientBoosting
5. Gaussian Naive Bayes
6. Decision Tree
7. Bagging Classifier
8. AdaBoost
9. K-Nearest Neighbors (KNN)
10. CatBoost
11. Random Forest
12. Extra Trees
13. Gradient Boosting

## Preprocessing Configurations

For microbiome-based models, three preprocessing options are considered:

- **SMOTETomek:** resampling method to address class imbalance.
- **PCA (Principal Component Analysis):** dimensionality reduction configured to retain 99% of the variance in the training data.
- **UMAP (Uniform Manifold Approximation and Projection):** dimensionality reduction to three components.

These options are represented by three Boolean parameters: `use_resampling`, `use_pca`, and `use_umap`.

The eight possible configurations are:

| SMOTETomek | PCA | UMAP |
|---|---|---|
| Disabled | Disabled | Disabled |
| Enabled | Disabled | Disabled |
| Disabled | Enabled | Disabled |
| Disabled | Disabled | Enabled |
| Enabled | Enabled | Disabled |
| Enabled | Disabled | Enabled |
| Disabled | Enabled | Enabled |
| Enabled | Enabled | Enabled |

When enabled, resampling is applied to the training data. PCA and UMAP are fitted using training data and subsequently applied to the corresponding test data.

**The eight preprocessing configurations apply to the microbiome-based analyses.** The sex-only scripts use a single configuration with all three options disabled.

## Validation Strategies

Four validation strategies are implemented.

### 1. Leave-One-Out Cross-Validation (LOO)

Leave-One-Out Cross-Validation trains a model on all participants except one and evaluates it on the participant left out. This process is repeated until each participant has been used as a test observation.

Performance metrics are calculated from the aggregated out-of-sample predictions.

In the sex-only LOO script, 2,000 bootstrap resamples of the aggregated predictions are used to estimate 95% percentile confidence intervals. This procedure resamples the predictions and does not retrain the model for every bootstrap replicate.

### 2. Stratified K-Fold Cross-Validation (SKF)

Stratified K-Fold Cross-Validation divides the dataset into 10 folds while approximately preserving the class proportions in each fold.

The model is trained on nine folds and evaluated on the remaining fold. This process is repeated until every fold has served as the test set.

The scripts use 10 folds, shuffling, and `random_state=42`. Performance is summarized using the mean and standard deviation across folds.

### 3. Repeated Stratified K-Fold Cross-Validation (RSKF)

Repeated Stratified K-Fold Cross-Validation repeats the 10-fold stratified procedure three times, resulting in 30 fold evaluations.

The scripts use `n_splits=10`, `n_repeats=3`, and `random_state=42`. Performance is summarized using the mean and standard deviation across the evaluations.

### 4. Train-Test Split (TTS)

The Train-Test Split strategy partitions the dataset into a training set containing 80% of the observations and a test set containing the remaining 20%.

The scripts use `test_size=0.2` and `random_state=42`.

In the sex-only TTS script, 2,000 subsamples containing 90% of the test observations, sampled without replacement, are used to estimate 95% percentile confidence intervals. These intervals reflect resampling of test observations and do not involve repeated model training.

## Evaluation Metrics

Model performance is assessed using the following metrics:

- **Accuracy:** proportion of correctly classified observations.
- **Balanced accuracy:** average recall across the two classes.
- **Recall (sensitivity):** proportion of mood disorder cases correctly identified.
- **Precision:** proportion of predicted mood disorder cases that are correctly classified.
- **F1-score:** harmonic mean of precision and recall.
- **ROC-AUC:** area under the receiver operating characteristic curve, when suitable predicted probabilities or scores are available.

The F1-score is the primary metric for model comparison. Balanced accuracy and the other metrics provide complementary information, particularly when the classes are imbalanced.

## Scripts

The repository contains 13 Python scripts organized around the four validation strategies and three predictor configurations.

### Microbiome + Sex

These scripts evaluate models using microbiome features combined with sex.

| Script | Validation strategy |
|---|---|
| `ml_r1_th_loo.py` | Leave-One-Out Cross-Validation |
| `ml_r1_th_rskf.py` | Repeated Stratified K-Fold |
| `ml_r1_th_skf.py` | Stratified K-Fold |
| `ml_r1_th_tts.py` | Train-Test Split |

### Microbiome Only

These scripts evaluate models using microbiome features without sex as a predictor.

| Script | Validation strategy |
|---|---|
| `ml_r1_th_loo_nosex.py` | Leave-One-Out Cross-Validation |
| `ml_r1_th_rskf_nosex.py` | Repeated Stratified K-Fold |
| `ml_r1_th_skf_nosex.py` | Stratified K-Fold |
| `ml_r1_th_tts_nosex.py` | Train-Test Split |

### Sex Only

These scripts evaluate models using sex as the sole predictor.

| Script | Validation strategy | Output file |
|---|---|---|
| `ml_r2_th_loo_sexonly.py` | Leave-One-Out Cross-Validation | `df_results_LOO_sex_only.csv` |
| `ml_r2_th_rskf_sexonly.py` | Repeated Stratified K-Fold | `df_results_RSKF_SexOnly.csv` |
| `ml_r2_th_skf_sexonly.py` | Stratified K-Fold | `df_results_SKF_SexOnly.csv` |
| `ml_r2_th_tts_sexonly.py` | Train-Test Split | `df_results_TTS_SexOnly.csv` |

The sex-only scripts use the same 13 algorithm families and the same TH-versus-control classification target. They disable SMOTETomek, PCA, and UMAP.

### ROC Curve Visualization

- `r1_curva_roc_revista.py`: script for generating ROC curve visualizations from model results.

## Results and Outputs

The analyses generate CSV files containing model performance metrics. Depending on the script, the results may include the validation strategy, algorithm, preprocessing configuration, performance metrics, and uncertainty estimates.

The sex-only scripts generate the following result files:

- `df_results_LOO_sex_only.csv`
- `df_results_RSKF_SexOnly.csv`
- `df_results_SKF_SexOnly.csv`
- `df_results_TTS_SexOnly.csv`

The microbiome-based scripts also generate results for their corresponding validation strategies and preprocessing configurations. The exact output filenames should be checked in the individual scripts.

The ROC curve visualization script can be used to compare model discrimination across the evaluated models.

## Reproducibility

Random seeds are specified in the scripts to improve reproducibility. The analyses use `random_state=42` for relevant data splits and cross-validation procedures. The sex-only LOO and TTS scripts also specify seeds for their bootstrap or subsampling procedures.

To reproduce the analyses:

1. Use the intended version of `table.csv`.
2. Ensure that the input file is available in the working directory expected by the scripts.
3. Install the Python libraries required by the selected script.
4. Run the script corresponding to the desired predictor configuration and validation strategy.
5. Preserve the generated CSV files and document any changes to the data or script settings.

Results may differ if the dataset, software versions, variable coding, or analysis settings change.

## Interpretation and Limitations

The sex-only models provide a baseline for comparison with models that use microbiome features. Differences in performance can help contextualize the classification results, but differences alone do not establish a statistically significant incremental contribution of microbiome features.

Formal comparisons between predictor configurations and consideration of uncertainty are needed to support conclusions about incremental predictive value.

Performance estimates also depend on sample size, class distribution, data quality, and validation design. The results should be interpreted as estimates of classification performance rather than evidence of clinical diagnostic utility.
