# -*- coding: utf-8 -*-
!pip install -q umap-learn catboost
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from tqdm.notebook import tqdm

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.decomposition import PCA

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score
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

from catboost import CatBoostClassifier

# ============================================================
# LOAD AND PREPARE DATA
# ============================================================

df = pd.read_csv("table.csv")

df["mood"] = df["mood"].map({
    0: "controle",
    2: "TH",
    3: "TH"
})

df = df.drop(["id"], axis=1)

# Microbiome ASVs
X_asv = df.drop(["mood", "sex"], axis=1)

# Outcome
y = df["mood"] == "TH"

# ============================================================
# RELATIVE ABUNDANCE
# ============================================================

X_asv_relative = (
    X_asv.T / X_asv.sum(axis=1)
).T


# ============================================================
# MICROBIOME-ONLY FEATURES
# ============================================================

X_features = X_asv_relative.copy()

# No sex added this time. It's in X_asv but doesn't enter X_features so it's not used

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
    (True,  True,  True)

]

# ============================================================
# REPEATED STRATIFIED K-FOLD
# ============================================================

N_SPLITS = 10
N_REPEATS = 3

rskf = RepeatedStratifiedKFold(
    n_splits=N_SPLITS,
    n_repeats=N_REPEATS,
    random_state=42
)


# ============================================================
# RESULTS
# ============================================================

results = []


# ============================================================
# MAIN LOOP
# ============================================================

for model_name, model in models.items():

    for use_resampling, use_pca, use_umap in configs:

        # ----------------------------------------------------
        # Store metrics from each fold/repetition
        # ----------------------------------------------------

        fold_metrics = {
            "accuracy": [],
            "balanced_accuracy": [],
            "recall": [],
            "precision": [],
            "f1": [],
            "roc_auc": []
        }


        # ----------------------------------------------------
        # RSKF
        # ----------------------------------------------------

        for fold_number, (train_index, test_index) in enumerate(
            tqdm(
                rskf.split(X_features, y),
                total=rskf.get_n_splits(X_features, y),
                desc=(
                    f"{model_name} RSKF | "
                    f"PCA={use_pca} | "
                    f"UMAP={use_umap} | "
                    f"SMOTE={use_resampling}"
                )
            ),
            start=1
        ):

            X_train = X_features.iloc[train_index].copy()
            X_test = X_features.iloc[test_index].copy()

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
            # PREDICTIONS
            # ------------------------------------------------

            y_pred = model.predict(
                X_test
            )


            # ------------------------------------------------
            # METRICS
            # ------------------------------------------------

            fold_metrics["accuracy"].append(
                accuracy_score(
                    y_test,
                    y_pred
                )
            )

            fold_metrics["balanced_accuracy"].append(
                balanced_accuracy_score(
                    y_test,
                    y_pred
                )
            )

            fold_metrics["recall"].append(
                recall_score(
                    y_test,
                    y_pred,
                    zero_division=0
                )
            )

            fold_metrics["precision"].append(
                precision_score(
                    y_test,
                    y_pred,
                    zero_division=0
                )
            )

            fold_metrics["f1"].append(
                f1_score(
                    y_test,
                    y_pred,
                    zero_division=0
                )
            )


            # ------------------------------------------------
            # ROC-AUC
            # ------------------------------------------------

            try:

                y_pred_proba = model.predict_proba(
                    X_test
                )[:, 1]

                fold_metrics["roc_auc"].append(
                    roc_auc_score(
                        y_test,
                        y_pred_proba
                    )
                )

            except Exception:

                pass


        # ====================================================
        # MEAN + SD
        # ====================================================

        metrics = {

            "validation": "RepeatedStratifiedKFold",

            "model": model_name,

            "use_resampling": use_resampling,

            "use_pca": use_pca,

            "use_umap": use_umap,

            # -----------------------------------------------
            # Accuracy
            # -----------------------------------------------

            "accuracy": np.mean(
                fold_metrics["accuracy"]
            ),

            "accuracy_SD": np.std(
                fold_metrics["accuracy"],
                ddof=1
            ),

            # -----------------------------------------------
            # Balanced Accuracy
            # -----------------------------------------------

            "balanced_accuracy": np.mean(
                fold_metrics["balanced_accuracy"]
            ),

            "balanced_accuracy_SD": np.std(
                fold_metrics["balanced_accuracy"],
                ddof=1
            ),

            # -----------------------------------------------
            # Recall
            # -----------------------------------------------

            "recall": np.mean(
                fold_metrics["recall"]
            ),

            "recall_SD": np.std(
                fold_metrics["recall"],
                ddof=1
            ),

            # -----------------------------------------------
            # Precision
            # -----------------------------------------------

            "precision": np.mean(
                fold_metrics["precision"]
            ),

            "precision_SD": np.std(
                fold_metrics["precision"],
                ddof=1
            ),

            # -----------------------------------------------
            # F1
            # -----------------------------------------------

            "f1": np.mean(
                fold_metrics["f1"]
            ),

            "f1_SD": np.std(
                fold_metrics["f1"],
                ddof=1
            ),

            # -----------------------------------------------
            # ROC-AUC
            # -----------------------------------------------

            "roc_auc": (
                np.mean(fold_metrics["roc_auc"])
                if len(fold_metrics["roc_auc"]) > 0
                else np.nan
            ),

            "roc_auc_SD": (
                np.std(
                    fold_metrics["roc_auc"],
                    ddof=1
                )
                if len(fold_metrics["roc_auc"]) > 1
                else np.nan
            ),

            # -----------------------------------------------
            # Number of folds/repetitions
            # -----------------------------------------------

            "n_splits": N_SPLITS,

            "n_repeats": N_REPEATS,

            "n_folds_total": (
                N_SPLITS * N_REPEATS
            )
        }


        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        results.append(
            metrics
        )


# ============================================================
# FINAL DATAFRAME
# ============================================================

df_results_RSKF = pd.DataFrame(
    results
)

# ============================================================
# SAVE CSV
# ============================================================

df_results_RSKF.to_csv(
    "df_results_RSKF_mean_SD.csv",
    index=False
)

# ============================================================
# DISPLAY
# ============================================================

print(
    df_results_RSKF.head()
)

print(
    "\nResults saved to:"
    " df_results_RSKF_mean_SD.csv"
)

print(
    "\nNumber of model/configuration combinations:",
    len(df_results_RSKF)
)