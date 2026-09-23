# -*- coding: utf-8 -*-
!pip install umap-learn plotly catboost

from catboost import CatBoostClassifier

from sklearn.model_selection import LeaveOneOut
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
# BOOTSTRAP CI FOR AGGREGATED LOO PREDICTIONS
# ============================================================

def calculate_loo_bootstrap_ci(
    y_true,
    y_pred,
    y_proba=None,
    n_bootstrap=2000,
    random_state=123
):
    """
    Calculate 95% confidence intervals by bootstrap resampling
    the aggregated individual predictions from Leave-One-Out CV.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_proba is not None:
        y_proba = np.asarray(y_proba)

    n_samples = len(y_true)

    rng = np.random.default_rng(random_state)

    accuracy_values = []
    balanced_accuracy_values = []
    recall_values = []
    precision_values = []
    f1_values = []
    roc_auc_values = []

    for _ in range(n_bootstrap):

        # ----------------------------------------------------
        # Resample the 93 LOO predictions WITH replacement
        # ----------------------------------------------------

        indices = rng.choice(
            n_samples,
            size=n_samples,
            replace=True
        )

        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]

        # ----------------------------------------------------
        # Accuracy
        # ----------------------------------------------------

        accuracy_values.append(
            accuracy_score(
                y_true_boot,
                y_pred_boot
            )
        )

        # ----------------------------------------------------
        # Balanced Accuracy
        # ----------------------------------------------------

        balanced_accuracy_values.append(
            balanced_accuracy_score(
                y_true_boot,
                y_pred_boot
            )
        )

        # ----------------------------------------------------
        # Recall
        # ----------------------------------------------------

        recall_values.append(
            recall_score(
                y_true_boot,
                y_pred_boot,
                zero_division=0
            )
        )

        # ----------------------------------------------------
        # Precision
        # ----------------------------------------------------

        precision_values.append(
            precision_score(
                y_true_boot,
                y_pred_boot,
                zero_division=0
            )
        )

        # ----------------------------------------------------
        # F1
        # ----------------------------------------------------

        f1_values.append(
            f1_score(
                y_true_boot,
                y_pred_boot,
                zero_division=0
            )
        )

        # ----------------------------------------------------
        # ROC-AUC
        # ----------------------------------------------------

        if (
            y_proba is not None
            and len(np.unique(y_true_boot)) == 2
        ):

            roc_auc_values.append(
                roc_auc_score(
                    y_true_boot,
                    y_proba[indices]
                )
            )

    # ========================================================
    # 95% CONFIDENCE INTERVALS
    # ========================================================

    ci_results = {

        "accuracy_CI_low": np.percentile(
            accuracy_values,
            2.5
        ),

        "accuracy_CI_high": np.percentile(
            accuracy_values,
            97.5
        ),

        "balanced_accuracy_CI_low": np.percentile(
            balanced_accuracy_values,
            2.5
        ),

        "balanced_accuracy_CI_high": np.percentile(
            balanced_accuracy_values,
            97.5
        ),

        "recall_CI_low": np.percentile(
            recall_values,
            2.5
        ),

        "recall_CI_high": np.percentile(
            recall_values,
            97.5
        ),

        "precision_CI_low": np.percentile(
            precision_values,
            2.5
        ),

        "precision_CI_high": np.percentile(
            precision_values,
            97.5
        ),

        "f1_CI_low": np.percentile(
            f1_values,
            2.5
        ),

        "f1_CI_high": np.percentile(
            f1_values,
            97.5
        )
    }

    # --------------------------------------------------------
    # ROC-AUC CI
    # --------------------------------------------------------

    if len(roc_auc_values) > 0:

        ci_results["roc_auc_CI_low"] = np.percentile(
            roc_auc_values,
            2.5
        )

        ci_results["roc_auc_CI_high"] = np.percentile(
            roc_auc_values,
            97.5
        )

    else:

        ci_results["roc_auc_CI_low"] = np.nan
        ci_results["roc_auc_CI_high"] = np.nan

    return ci_results

# ============================================================
# LOO
# ============================================================

loo = LeaveOneOut()

# Number of bootstrap repetitions
N_BOOTSTRAP = 2000

# Random seed for bootstrap
BOOTSTRAP_RANDOM_STATE = 123

results = []
for model_name, model in models.items():

    for use_resampling, use_pca, use_umap in configs:

        # ----------------------------------------------------
        # Store the 93 individual LOO predictions
        # ----------------------------------------------------

        y_tests = []
        y_preds = []
        y_preds_proba = []

        # ----------------------------------------------------
        # LEAVE-ONE-OUT
        # ----------------------------------------------------

        for train_index, test_index in tqdm(
            loo.split(X_features),
            total=loo.get_n_splits(X_features),
            desc=(
                f"{model_name} LOO, "
                f"PCA={use_pca}, "
                f"UMAP={use_umap}, "
                f"SMOTE={use_resampling}"
            )
        ):

            X_train = X_features.iloc[train_index]
            X_test = X_features.iloc[test_index]

            y_train = y.iloc[train_index]
            y_test = y.iloc[test_index]

            # ------------------------------------------------
            # 1. SMOTETOMEK
            # ------------------------------------------------

            if use_resampling:

                smote_tomek = SMOTETomek(
                    random_state=42
                )

                X_train, y_train = smote_tomek.fit_resample(
                    X_train,
                    y_train
                )

            # ------------------------------------------------
            # 2. PCA
            # ------------------------------------------------

            if use_pca:

                pca = PCA(
                    n_components=0.99,
                    random_state=42
                )

                X_train = pca.fit_transform(
                    X_train
                )

                X_test = pca.transform(
                    X_test
                )

            # ------------------------------------------------
            # 3. UMAP
            # ------------------------------------------------

            if use_umap:

                umap = UMAP(
                    n_components=3,
                    random_state=42
                )

                X_train = umap.fit_transform(
                    X_train,
                    y_train
                )

                X_test = umap.transform(
                    X_test
                )

            # ------------------------------------------------
            # TRAIN MODEL
            # ------------------------------------------------

            model.fit(
                X_train,
                y_train
            )

            # ------------------------------------------------
            # PREDICTION
            # ------------------------------------------------

            y_pred = model.predict(
                X_test
            )

            y_tests.extend(
                y_test
            )

            y_preds.extend(
                y_pred
            )

            # ------------------------------------------------
            # PREDICTED PROBABILITY
            # ------------------------------------------------

            try:

                y_pred_proba = model.predict_proba(
                    X_test
                )[:, 1]

                y_preds_proba.extend(
                    y_pred_proba
                )

            except Exception:

                pass

        # ====================================================
        # AGGREGATED LOO PERFORMANCE
        # ====================================================

        metrics = {

            "validation": "LOO",

            "model": model_name,

            "use_resampling": use_resampling,

            "use_pca": use_pca,

            "use_umap": use_umap,

            "accuracy": accuracy_score(
                y_tests,
                y_preds
            ),

            "balanced_accuracy": balanced_accuracy_score(
                y_tests,
                y_preds
            ),

            "recall": recall_score(
                y_tests,
                y_preds,
                zero_division=0
            ),

            "precision": precision_score(
                y_tests,
                y_preds,
                zero_division=0
            ),

            "f1": f1_score(
                y_tests,
                y_preds,
                zero_division=0
            )
        }

        # ====================================================
        # ROC-AUC
        # ====================================================

        if len(y_preds_proba) == len(y_tests):

            metrics["roc_auc"] = roc_auc_score(
                y_tests,
                y_preds_proba
            )

        else:

            metrics["roc_auc"] = None

        # ====================================================
        # BOOTSTRAP 95% CI
        # ====================================================

        ci_results = calculate_loo_bootstrap_ci(

            y_true=y_tests,

            y_pred=y_preds,

            y_proba=(
                y_preds_proba
                if len(y_preds_proba) == len(y_tests)
                else None
            ),

            n_bootstrap=N_BOOTSTRAP,

            random_state=BOOTSTRAP_RANDOM_STATE
        )

        # ====================================================
        # ADD CI TO RESULTS
        # ====================================================

        metrics.update(
            ci_results
        )

        # ====================================================
        # SAVE BOOTSTRAP INFORMATION
        # ====================================================

        metrics["n_loo_predictions"] = len(y_tests)

        metrics["n_bootstrap"] = N_BOOTSTRAP

        metrics["CI_level"] = 0.95

        # ====================================================
        # SAVE RESULT
        # ====================================================

        results.append(
            metrics
        )


# ============================================================
# FINAL DATAFRAME
# ============================================================

df_results = pd.DataFrame(
    results
)

print(
    df_results[
        df_results["validation"] == "LOO"
    ].head()
)

df_results.to_csv(
    "df_results_LOO.csv",
    index=False
)

print("Arquivo salvo como df_results_LOO.csv")