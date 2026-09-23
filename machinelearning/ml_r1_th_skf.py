# -*- coding: utf-8 -*-
!pip install -q umap-learn catboost

import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import StratifiedKFold
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

from catboost import CatBoostClassifier

from imblearn.combine import SMOTETomek

from umap import UMAP

from tqdm.notebook import tqdm

# ============================================================
# LOAD AND PREPARE DATA
# ============================================================

df = pd.read_csv("table.csv")

df["mood"] = df["mood"].map({
    0: "controle",
    2: "TH",
    3: "TH"
})

df = df.drop(columns=["id"])

# Microbiome ASVs only
X_asv = df.drop(columns=["mood", "sex"])

# Outcome
y = df["mood"] == "TH"

# ============================================================
# RELATIVE ABUNDANCE
# ============================================================

X_asv_relative = (
    X_asv.T / X_asv.sum(axis=1)
).T


# ============================================================
# COMBINED FEATURES: ASVs + SEX
# ============================================================

X_features = X_asv_relative.copy()

X_features["sex"] = df["sex"] == 1


warnings.filterwarnings("ignore")

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

    (True, False, False),
    (True, False, True),

    (False, True, False),
    (False, True, True),

    (True, True, False),
    (True, True, True)
]

# ============================================================
# STRATIFIED K-FOLD
# ============================================================

N_SPLITS = 10

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
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
        # Store metrics for EACH fold
        # ----------------------------------------------------

        fold_accuracy = []
        fold_balanced_accuracy = []
        fold_recall = []
        fold_precision = []
        fold_f1 = []
        fold_roc_auc = []


        # ----------------------------------------------------
        # STRATIFIED K-FOLD
        # ----------------------------------------------------

        for train_index, test_index in tqdm(
            skf.split(X_features, y),
            total=N_SPLITS,
            desc=(
                f"{model_name} | "
                f"PCA={use_pca} | "
                f"UMAP={use_umap} | "
                f"SMOTE={use_resampling}"
            )
        ):

            # ------------------------------------------------
            # TRAIN / TEST
            # ------------------------------------------------

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
            # PROBABILITIES
            # ------------------------------------------------

            y_pred_proba = None

            try:

                y_pred_proba = model.predict_proba(
                    X_test
                )[:, 1]

            except Exception:

                pass


            # ------------------------------------------------
            # METRICS FOR THIS FOLD
            # ------------------------------------------------

            fold_accuracy.append(
                accuracy_score(
                    y_test,
                    y_pred
                )
            )

            fold_balanced_accuracy.append(
                balanced_accuracy_score(
                    y_test,
                    y_pred
                )
            )

            fold_recall.append(
                recall_score(
                    y_test,
                    y_pred,
                    zero_division=0
                )
            )

            fold_precision.append(
                precision_score(
                    y_test,
                    y_pred,
                    zero_division=0
                )
            )

            fold_f1.append(
                f1_score(
                    y_test,
                    y_pred,
                    zero_division=0
                )
            )


            # ------------------------------------------------
            # ROC-AUC
            # ------------------------------------------------

            if (
                y_pred_proba is not None
                and len(np.unique(y_test)) == 2
            ):

                fold_roc_auc.append(
                    roc_auc_score(
                        y_test,
                        y_pred_proba
                    )
                )


        # ====================================================
        # MEAN AND STANDARD DEVIATION
        # ====================================================

        metrics = {

            "validation": "StratifiedKFold",

            "model": model_name,

            "use_resampling": use_resampling,

            "use_pca": use_pca,

            "use_umap": use_umap,


            # ------------------------------------------------
            # MEAN
            # ------------------------------------------------

            "accuracy": np.mean(
                fold_accuracy
            ),

            "balanced_accuracy": np.mean(
                fold_balanced_accuracy
            ),

            "recall": np.mean(
                fold_recall
            ),

            "precision": np.mean(
                fold_precision
            ),

            "f1": np.mean(
                fold_f1
            ),

            "roc_auc": (
                np.mean(fold_roc_auc)
                if len(fold_roc_auc) > 0
                else np.nan
            ),


            # ------------------------------------------------
            # STANDARD DEVIATION
            # ------------------------------------------------

            "accuracy_SD": np.std(
                fold_accuracy,
                ddof=1
            ),

            "balanced_accuracy_SD": np.std(
                fold_balanced_accuracy,
                ddof=1
            ),

            "recall_SD": np.std(
                fold_recall,
                ddof=1
            ),

            "precision_SD": np.std(
                fold_precision,
                ddof=1
            ),

            "f1_SD": np.std(
                fold_f1,
                ddof=1
            ),

            "roc_auc_SD": (
                np.std(
                    fold_roc_auc,
                    ddof=1
                )
                if len(fold_roc_auc) > 1
                else np.nan
            )
        }


        # ----------------------------------------------------
        # SAVE NUMBER OF FOLDS
        # ----------------------------------------------------

        metrics["n_folds"] = N_SPLITS


        # ----------------------------------------------------
        # SAVE RESULTS
        # ----------------------------------------------------

        results.append(
            metrics
        )

# ============================================================
# FINAL DATAFRAME
# ============================================================

df_results = pd.DataFrame(
    results
)


# ============================================================
# SAVE CSV
# ============================================================

df_results.to_csv(
    "df_results_StratifiedKFold.csv",
    index=False
)

# ============================================================
# DISPLAY
# ============================================================

display(df_results)

print(
    f"\nResults saved to df_results_StratifiedKFold.csv"
)

print(
    f"Number of model/configuration combinations: "
    f"{len(df_results)}"
)