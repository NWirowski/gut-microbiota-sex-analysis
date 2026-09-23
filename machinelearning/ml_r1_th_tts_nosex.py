# -*- coding: utf-8 -*-
!pip install umap-learn plotly catboost

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.decomposition import PCA
from sklearn.metrics import (
    balanced_accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    accuracy_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from imblearn.combine import SMOTETomek
from umap import UMAP

from tqdm.notebook import tqdm
from plotly import express as ex

import pandas as pd
import numpy as np
import warnings

# ============================================================
# LOAD AND PREPARE DATA
# ============================================================

df = pd.read_csv('table.csv')

df['mood'] = df['mood'].map(
    lambda x: {
        0: "controle",
        2: "TH",
        3: "TH"
    }[x]
)

df = df.drop(['id'], axis=1)

X_asv = df.drop(['mood', 'sex'], axis=1)

y = df['mood'] == 'TH'

# ============================================================
# RELATIVE ABUNDANCE
# ============================================================

X_asv_relative = (X_asv.T / X_asv.sum(axis=1)).T

# ============================================================
# MICROBIOME-ONLY FEATURES
# ============================================================

X_features = X_asv_relative.copy()

# No sex added this time. It's in X_asv but doesn't enter X_features so it's not used

# ============================================================
# TRAIN-TEST SPLIT
# ============================================================

from umap import umap_

warnings.filterwarnings(
    "ignore",
    message="n_jobs value 1 overridden to 1 by setting random_state.*",
    category=UserWarning
)

# ============================================================
# MODELS
# ============================================================

base_tree = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

models = {
    "LogisticRegression": LogisticRegression(
        max_iter=200,
        random_state=42
    ),

    "SVC": SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    ),

    "MLP": MLPClassifier(
        hidden_layer_sizes=(100,),
        max_iter=2000,
        random_state=42
    ),

    "HistGradientBoosting": HistGradientBoostingClassifier(
        random_state=42
    ),

    "GaussianNB": GaussianNB(),

    "DecisionTree": DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
        random_state=42
    ),

    "Bagging": BaggingClassifier(
        estimator=base_tree,
        n_estimators=50,
        random_state=42
    ),

    "AdaBoost": AdaBoostClassifier(
        n_estimators=100,
        learning_rate=0.5,
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5,
        weights="distance"
    ),

    "CatBoost": CatBoostClassifier(
        verbose=False,
        random_state=42
    ),

    "RandomForest": RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42
    ),

    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42
    ),

    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=2,
        random_state=42
    )
}

# ============================================================
# PREPROCESSING CONFIGURATIONS
# ============================================================

configs = [
    (False, False, False),
    (False, False, True),
    (True,  False, False),
    (True,  False, True),
    (False, True,  False),
    (False, True,  True),
    (True,  True,  False),
    (True,  True,  True),
]

# ============================================================
# SUBSAMPLING PARAMETERS
# ============================================================

N_SUBSAMPLES = 2000
SUBSAMPLE_FRACTION = 0.90
CI_LEVEL = 0.95

# Separate random seed for the subsampling procedure
SUBSAMPLING_RANDOM_STATE = 123

# ============================================================
# FUNCTION TO CALCULATE 95% CI USING 90% TEST-SET SUBSAMPLING
# ============================================================

def calculate_subsampling_ci(
    y_true,
    y_pred,
    y_proba=None,
    n_subsamples=2000,
    sample_fraction=0.90,
    random_state=123
):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_proba is not None:
        y_proba = np.asarray(y_proba)

    n_test = len(y_true)
    sample_size = int(np.ceil(n_test * sample_fraction))

    rng = np.random.default_rng(random_state)

    accuracy_values = []
    balanced_accuracy_values = []
    recall_values = []
    precision_values = []
    f1_values = []
    roc_auc_values = []

    for _ in range(n_subsamples):

        indices = rng.choice(
            n_test,
            size=sample_size,
            replace=False
        )

        y_true_subsample = y_true[indices]
        y_pred_subsample = y_pred[indices]

        accuracy_values.append(
            accuracy_score(y_true_subsample, y_pred_subsample)
        )

        balanced_accuracy_values.append(
            balanced_accuracy_score(
                y_true_subsample,
                y_pred_subsample
            )
        )

        recall_values.append(
            recall_score(
                y_true_subsample,
                y_pred_subsample,
                zero_division=0
            )
        )

        precision_values.append(
            precision_score(
                y_true_subsample,
                y_pred_subsample,
                zero_division=0
            )
        )

        f1_values.append(
            f1_score(
                y_true_subsample,
                y_pred_subsample,
                zero_division=0
            )
        )

        if (
            y_proba is not None
            and len(np.unique(y_true_subsample)) == 2
        ):
            roc_auc_values.append(
                roc_auc_score(
                    y_true_subsample,
                    y_proba[indices]
                )
            )

    ci_results = {

        "accuracy_CI_low": np.percentile(
            accuracy_values, 2.5
        ),

        "accuracy_CI_high": np.percentile(
            accuracy_values, 97.5
        ),

        "balanced_accuracy_CI_low": np.percentile(
            balanced_accuracy_values, 2.5
        ),

        "balanced_accuracy_CI_high": np.percentile(
            balanced_accuracy_values, 97.5
        ),

        "recall_CI_low": np.percentile(
            recall_values, 2.5
        ),

        "recall_CI_high": np.percentile(
            recall_values, 97.5
        ),

        "precision_CI_low": np.percentile(
            precision_values, 2.5
        ),

        "precision_CI_high": np.percentile(
            precision_values, 97.5
        ),

        "f1_CI_low": np.percentile(
            f1_values, 2.5
        ),

        "f1_CI_high": np.percentile(
            f1_values, 97.5
        )
    }

    if len(roc_auc_values) > 0:

        ci_results["roc_auc_CI_low"] = np.percentile(
            roc_auc_values, 2.5
        )

        ci_results["roc_auc_CI_high"] = np.percentile(
            roc_auc_values, 97.5
        )

    else:

        ci_results["roc_auc_CI_low"] = np.nan
        ci_results["roc_auc_CI_high"] = np.nan

    return ci_results


# ============================================================
# MAIN LOOP
# ============================================================

results = []

for model_name, model in tqdm(
    models.items(),
    desc="Models"
):

    for use_resampling, use_pca, use_umap in configs:

        # ----------------------------------------------------
        # TRAIN-TEST SPLIT
        # ----------------------------------------------------

        X_train, X_test, y_train, y_test = train_test_split(
            X_features,
            y,
            test_size=0.2,
            random_state=42
        )

        # ----------------------------------------------------
        # 1. SMOTETOMEK
        # ----------------------------------------------------

        if use_resampling:

            smote_tomek = SMOTETomek(
                random_state=42
            )

            X_train, y_train = smote_tomek.fit_resample(
                X_train,
                y_train
            )

        # ----------------------------------------------------
        # 2. PCA
        # ----------------------------------------------------

        if use_pca:

            pca = PCA(
                n_components=0.99,
                random_state=42
            )

            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

        # ----------------------------------------------------
        # 3. UMAP
        # ----------------------------------------------------

        if use_umap:

            umap = UMAP(
                n_components=3,
                random_state=42
            )

            X_train = umap.fit_transform(
                X_train,
                y_train
            )

            X_test = umap.transform(X_test)

        # ----------------------------------------------------
        # TRAIN MODEL
        # ----------------------------------------------------

        model.fit(
            X_train,
            y_train
        )

        # ----------------------------------------------------
        # PREDICTIONS
        # ----------------------------------------------------

        y_preds = model.predict(X_test)

        y_preds_proba = None

        try:

            y_preds_proba = model.predict_proba(
                X_test
            )[:, 1]

        except Exception:

            pass

        # ----------------------------------------------------
        # ORIGINAL PERFORMANCE METRICS
        # ----------------------------------------------------

        metrics = {

            "validation": "TTS",

            "model": model_name,

            "use_resampling": use_resampling,

            "use_pca": use_pca,

            "use_umap": use_umap,

            "accuracy": accuracy_score(
                y_test,
                y_preds
            ),

            "balanced_accuracy": balanced_accuracy_score(
                y_test,
                y_preds
            ),

            "recall": recall_score(
                y_test,
                y_preds,
                zero_division=0
            ),

            "precision": precision_score(
                y_test,
                y_preds,
                zero_division=0
            ),

            "f1": f1_score(
                y_test,
                y_preds,
                zero_division=0
            ),
        }

        # ----------------------------------------------------
        # ROC-AUC
        # ----------------------------------------------------

        if y_preds_proba is not None:

            metrics["roc_auc"] = roc_auc_score(
                y_test,
                y_preds_proba
            )

        else:

            metrics["roc_auc"] = None

        # ----------------------------------------------------
        # 90% TEST-SET SUBSAMPLING
        # ----------------------------------------------------

        ci_results = calculate_subsampling_ci(

            y_true=y_test,

            y_pred=y_preds,

            y_proba=y_preds_proba,

            n_subsamples=N_SUBSAMPLES,

            sample_fraction=SUBSAMPLE_FRACTION,

            random_state=SUBSAMPLING_RANDOM_STATE
        )

        # ----------------------------------------------------
        # ADD CI RESULTS
        # ----------------------------------------------------

        metrics.update(ci_results)

        # ----------------------------------------------------
        # SAVE SUBSAMPLING INFORMATION
        # ----------------------------------------------------

        metrics["n_test"] = len(y_test)

        metrics["subsample_fraction"] = SUBSAMPLE_FRACTION

        metrics["subsample_size"] = int(
            np.ceil(
                len(y_test) * SUBSAMPLE_FRACTION
            )
        )

        metrics["n_subsamples"] = N_SUBSAMPLES

        metrics["CI_level"] = CI_LEVEL

        # ----------------------------------------------------
        # SAVE RESULT
        # ----------------------------------------------------

        results.append(metrics)


# ============================================================
# FINAL DATAFRAME
# ============================================================

df_results = pd.DataFrame(results)


# ============================================================
# SAVE CSV
# ============================================================

df_results.to_csv(
    "df_results.csv",
    index=False
)

print(df_results.head())

print(
    f"\nResults saved to df_results.csv"
)

print(
    f"Number of model/configuration combinations: "
    f"{len(df_results)}"
)