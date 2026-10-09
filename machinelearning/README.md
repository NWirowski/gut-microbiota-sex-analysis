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

Three predictor sets are evaluated in the revised binary classification analysis.

### Combined model

The **combined model** uses:

- 2,496 microbiome ASVs
- Sex

The feature matrix combines the microbiome profile with sex as an additional predictor.

### Microbiome-only model

The **microbiome-only model** uses:

- 2,496 microbiome ASVs

Sex is excluded from the feature matrix. These analyses are implemented in scripts with the `_nosex` suffix.

### Sex-only model

The **sex-only model** uses:

- Sex as the sole predictor

These models provide a baseline for evaluating how well sex alone distinguishes participants with Mood Disorder from Healthy Controls. They do not use microbiome ASVs and do not apply SMOTETomek, PCA, or UMAP.

The sex-only analyses are implemented in scripts with the `_sexonly` suffix.

| Predictor set | Features |
|---|---|
| Combined model | 2,496 ASVs + sex |
| Microbiome-only model | 2,496 ASVs |
| Sex-only model | Sex only |

Comparing these predictor sets helps characterize classification performance using the microbiome, sex, or both.

## Preprocessing Configurations

For the microbiome-based analyses, the algorithms are evaluated using preprocessing configurations based on three components:

- **SMOTETomek** — class-imbalance handling
- **PCA** — linear dimensionality reduction
- **UMAP** — nonlinear dimensionality reduction

The eight possible combinations are:

| SMOTETomek | PCA | UMAP |
|---|---|---|
| No | No | No |
| No | No | Yes |
| Yes | No | No |
| Yes | No | Yes |
| No | Yes | No |
| No | Yes | Yes |
| Yes | Yes | No |
| Yes | Yes | Yes |

**Important:** The `_sexonly` scripts use only the configuration with all three components disabled. Therefore, they evaluate the algorithms using sex alone, without resampling or dimensionality reduction. The eight configurations listed above do not apply to these scripts.

## Validation Strategies

Four validation strategies are implemented across the machine learning analyses: Leave-One-Out Cross-Validation (LOO), Stratified K-Fold Cross-Validation (SKF), Repeated Stratified K-Fold Cross-Validation (RSKF), and Train-Test Split (TTS).

### Leave-One-Out Cross-Validation (LOO)

The LOO analyses train each model on all but one participant and use the held-out participant for prediction. This process is repeated until every participant has been held out once.

The sex-only implementation is:

- `ml_r2_th_loo_sexonly.py`

The script calculates performance metrics from the aggregated individual-level LOO predictions. It estimates 95% percentile confidence intervals using 2,000 bootstrap resamples of these predictions, sampled with replacement.

The reported metrics are:

- Accuracy
- Balanced accuracy
- Recall
- Precision
- F1-score
- ROC-AUC

### Stratified K-Fold Cross-Validation (SKF)

The SKF analyses use **10 folds**, preserving the class distribution across folds. Performance is calculated separately for each fold and summarized using the mean and standard deviation (SD).

The sex-only implementation is:

- `ml_r2_th_skf_sexonly.py`

The reported metrics are accuracy, balanced accuracy, recall, precision, F1-score, and ROC-AUC.

### Repeated Stratified K-Fold Cross-Validation (RSKF)

The RSKF analyses use:

- 10 folds
- 3 repetitions
- 30 fold evaluations in total

Performance is calculated for each fold and repetition and summarized using the mean and SD.

The sex-only implementation is:

- `ml_r2_th_rskf_sexonly.py`

The reported metrics are accuracy, balanced accuracy, recall, precision, F1-score, and ROC-AUC.

### Train-Test Split (TTS)

The TTS analyses use a fixed **80% training / 20% test split**, with `random_state = 42`.

The sex-only implementation is:

- `ml_r2_th_tts_sexonly.py`

Performance is evaluated on the held-out test set. To estimate uncertainty, the script generates 2,000 subsamples containing 90% of the test observations, sampled without replacement, and calculates 95% percentile confidence intervals.

The reported metrics are accuracy, balanced accuracy, recall, precision, F1-score, and ROC-AUC.

## Script Overview

The scripts below implement the sex-only baseline for the revised Mood Disorder versus Control classification.

| Script | Predictor set | Validation strategy | Output |
|---|---|---|---|
| `ml_r2_th_loo_sexonly.py` | Sex only | Leave-One-Out CV | `df_results_LOO_sex_only.csv` |
| `ml_r2_th_skf_sexonly.py` | Sex only | 10-fold Stratified CV | `df_results_SKF_SexOnly.csv` |
| `ml_r2_th_rskf_sexonly.py` | Sex only | 10-fold × 3-repetition Stratified CV | `df_results_RSKF_SexOnly.csv` |
| `ml_r2_th_tts_sexonly.py` | Sex only | 80/20 Train-Test Split | `df_results_TTS_SexOnly.csv` |

The existing combined and microbiome-only scripts remain documented separately. Their predictor sets and preprocessing configurations should not be inferred from the sex-only scripts.

## Sex-Only Baseline Analysis

The sex-only models use the same 13 algorithm families as the other classification analyses:

- Logistic Regression
- Support Vector Classifier (SVC)
- Multilayer Perceptron (MLP)
- HistGradientBoosting
- Gaussian Naive Bayes
- Decision Tree
- Bagging
- AdaBoost
- K-Nearest Neighbors (KNN)
- CatBoost
- Random Forest
- Extra Trees
- Gradient Boosting

Although the same algorithm families are evaluated, the sex-only feature matrix contains a single binary predictor. PCA and UMAP are disabled, and SMOTETomek is not applied.

The sex-only analysis serves as a baseline for contextualizing the predictive performance of models that include microbiome features. Differences in performance alone do not establish that the microbiome provides a statistically significant incremental contribution; that conclusion would require an appropriate formal comparison.

## Outputs

Each sex-only validation script saves a CSV file containing model performance results.

The output tables include the validation strategy, model name, preprocessing flags, and performance metrics. Depending on the validation strategy, they also include standard deviations, confidence intervals, and information about the number of folds, test observations, bootstrap resamples, or test-set subsamples.

The CSV output files are:

- `df_results_LOO_sex_only.csv`
- `df_results_SKF_SexOnly.csv`
- `df_results_RSKF_SexOnly.csv`
- `df_results_TTS_SexOnly.csv`

## Analysis Overview

| Component | Specification |
|---|---|
| Outcome | Mood Disorder vs Control |
| Predictor sets | Combined, microbiome-only, sex-only |
| Microbiome predictors | 2,496 ASVs |
| Sex-only predictors | One binary sex indicator |
| Machine learning algorithms | 13 |
| Validation strategies | LOO, SKF, RSKF, TTS |
| SKF | 10 folds; mean ± SD |
| RSKF | 10 folds × 3 repetitions; mean ± SD |
| LOO uncertainty | 2,000 bootstrap resamples of aggregated predictions |
| TTS | 80% training / 20% test |
| TTS uncertainty | 2,000 subsamples of 90% of the test set |
| Confidence intervals | 95% percentile intervals |
| Primary performance metric | F1-score |

## Reproducibility

The scripts use fixed random seeds where applicable, including `random_state = 42` for model fitting and validation procedures and `random_state = 123` for the LOO bootstrap and TTS subsampling procedures.

The analyses were developed for execution in Python/Google Colab. Required packages depend on the specific script and include `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`, `umap-learn`, `catboost`, and `tqdm`. Some scripts may import additional packages.

The input file `table.csv` must be available in the working directory. The file must contain the diagnostic outcome, sex, participant identifier (`id`), and, for microbiome-based analyses, the ASV features required by the corresponding script.
