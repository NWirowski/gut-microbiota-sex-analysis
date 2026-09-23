# -*- coding: utf-8 -*-
"""R1-Curva_ROC_revista.ipynb
Part 1 - Machine Learning Analysis
"""

!pip install -q umap-learn imbalanced-learn


import os
import pickle
import warnings

import pandas as pd
import numpy as np

from tqdm.notebook import tqdm

from sklearn.base import clone
from sklearn.model_selection import LeaveOneOut
from sklearn.decomposition import PCA

from sklearn.metrics import (
    balanced_accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    accuracy_score,
    roc_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from imblearn.combine import SMOTETomek

from umap import UMAP

# ============================================================
# LOAD AND PREPARE DATA
# ============================================================

df = pd.read_csv("table.csv")


df['mood'] = df['mood'].map(
    lambda x: {
        0: "controle",
        2: "TH",
        3: "TH"
    }[x]
)


df = df.drop(
    ['id'],
    axis=1
)


# ASV table
X_asv = df.drop(
    ['mood', 'sex'],
    axis=1
)


# Outcome
y = df['mood'] == 'TH'

# ============================================================
# RELATIVE ABUNDANCE
# ============================================================

X_asv_relative = (
    X_asv.T /
    X_asv.sum(axis=1)
).T

# ============================================================
# BASE FEATURES
#
# Microbiome only.
#
# Sex is added later only if use_sex=True.
# ============================================================

X_features_base = X_asv_relative.copy()


warnings.filterwarnings(
    "ignore",
    message=(
        "n_jobs value 1 overridden to 1 by "
        "setting random_state.*"
    ),
    category=UserWarning
)

# ============================================================
# MODELS - Same parameters as the original script.
# ============================================================

models = {

    "LogisticRegression": LogisticRegression(
        max_iter=200,
        random_state=42
    ),

    "SVC": SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    )
}

# ============================================================
# SELECTED CONFIGURATIONS
# ============================================================

selected_configs = [

    {
        "model": "LogisticRegression",
        "use_resampling": True,
        "use_pca": True,
        "use_umap": False
    },

    {
        "model": "LogisticRegression",
        "use_resampling": True,
        "use_pca": False,
        "use_umap": False
    },

    {
        "model": "SVC",
        "use_resampling": True,
        "use_pca": False,
        "use_umap": False
    },

    {
        "model": "SVC",
        "use_resampling": True,
        "use_pca": False,
        "use_umap": True
    },

    {
        "model": "SVC",
        "use_resampling": True,
        "use_pca": True,
        "use_umap": False
    }
]

# ============================================================
# FEATURES
# ============================================================

def get_features(use_sex):

    X_features = X_features_base.copy()

    if use_sex:

        X_features['sex'] = (
            df['sex'] == 1
        )

    return X_features

# ============================================================
# BOOTSTRAP CI
#
# 2,000 bootstrap samples
# WITH replacement
# 95% percentile CI
#
# Applied to the aggregated individual LOO predictions.
# ============================================================

def calculate_loo_bootstrap_ci(
    y_true,
    y_pred,
    y_proba=None,
    n_bootstrap=2000,
    random_state=123
):

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    if y_proba is not None:

        y_proba = np.asarray(
            y_proba
        )


    n_samples = len(
        y_true
    )


    rng = np.random.default_rng(
        random_state
    )


    accuracy_values = []
    balanced_accuracy_values = []
    recall_values = []
    precision_values = []
    f1_values = []
    roc_auc_values = []


    # --------------------------------------------------------
    # BOOTSTRAP
    # --------------------------------------------------------

    for _ in range(
        n_bootstrap
    ):

        indices = rng.choice(
            n_samples,
            size=n_samples,
            replace=True
        )


        y_true_boot = \
            y_true[indices]

        y_pred_boot = \
            y_pred[indices]


        # Accuracy
        accuracy_values.append(
            accuracy_score(
                y_true_boot,
                y_pred_boot
            )
        )


        # Balanced Accuracy
        balanced_accuracy_values.append(
            balanced_accuracy_score(
                y_true_boot,
                y_pred_boot
            )
        )


        # Recall
        recall_values.append(
            recall_score(
                y_true_boot,
                y_pred_boot,
                zero_division=0
            )
        )


        # Precision
        precision_values.append(
            precision_score(
                y_true_boot,
                y_pred_boot,
                zero_division=0
            )
        )


        # F1
        f1_values.append(
            f1_score(
                y_true_boot,
                y_pred_boot,
                zero_division=0
            )
        )


        # ROC AUC
        if (
            y_proba is not None
            and len(
                np.unique(
                    y_true_boot
                )
            ) == 2
        ):

            roc_auc_values.append(

                roc_auc_score(
                    y_true_boot,
                    y_proba[indices]
                )
            )


    # --------------------------------------------------------
    # 95% CI
    # --------------------------------------------------------

    ci_results = {

        "accuracy_CI_low":
            np.percentile(
                accuracy_values,
                2.5
            ),

        "accuracy_CI_high":
            np.percentile(
                accuracy_values,
                97.5
            ),

        "balanced_accuracy_CI_low":
            np.percentile(
                balanced_accuracy_values,
                2.5
            ),

        "balanced_accuracy_CI_high":
            np.percentile(
                balanced_accuracy_values,
                97.5
            ),

        "recall_CI_low":
            np.percentile(
                recall_values,
                2.5
            ),

        "recall_CI_high":
            np.percentile(
                recall_values,
                97.5
            ),

        "precision_CI_low":
            np.percentile(
                precision_values,
                2.5
            ),

        "precision_CI_high":
            np.percentile(
                precision_values,
                97.5
            ),

        "f1_CI_low":
            np.percentile(
                f1_values,
                2.5
            ),

        "f1_CI_high":
            np.percentile(
                f1_values,
                97.5
            )
    }


    # ROC AUC CI
    if len(
        roc_auc_values
    ) > 0:

        ci_results[
            "roc_auc_CI_low"
        ] = np.percentile(
            roc_auc_values,
            2.5
        )

        ci_results[
            "roc_auc_CI_high"
        ] = np.percentile(
            roc_auc_values,
            97.5
        )

    else:

        ci_results[
            "roc_auc_CI_low"
        ] = np.nan

        ci_results[
            "roc_auc_CI_high"
        ] = np.nan


    return ci_results


# ============================================================
# SETTINGS
# ============================================================

N_BOOTSTRAP = 2000

BOOTSTRAP_RANDOM_STATE = 123

loo = LeaveOneOut()


# ============================================================
# CHECKPOINT FILE
# ============================================================

CHECKPOINT_FILE = (
    "LOO_ROC_selected_results.pkl"
)


CSV_FILE = (
    "df_results_LOO_ROC_selected.csv"
)


# ============================================================
# LOAD EXISTING CHECKPOINT
# ============================================================

if os.path.exists(
    CHECKPOINT_FILE
):

    print(
        f"Existing checkpoint found: "
        f"{CHECKPOINT_FILE}"
    )

    with open(
        CHECKPOINT_FILE,
        "rb"
    ) as f:

        saved_results = pickle.load(
            f
        )

else:

    print(
        "No existing checkpoint found."
    )

    saved_results = []


# ============================================================
# IDENTIFY COMPLETED ANALYSES
# ============================================================

completed_keys = set()


for result in saved_results:

    key = (

        result["model"],

        result["use_sex"],

        result["use_resampling"],

        result["use_pca"],

        result["use_umap"]
    )

    completed_keys.add(
        key
    )


print(
    f"\nCompleted analyses already saved: "
    f"{len(completed_keys)}/10"
)


# ============================================================
# MAIN LOO LOOP
# ============================================================

for config in selected_configs:

    model_name = config[
        "model"
    ]

    use_resampling = config[
        "use_resampling"
    ]

    use_pca = config[
        "use_pca"
    ]

    use_umap = config[
        "use_umap"
    ]


    # --------------------------------------------------------
    # WITH SEX / NO SEX
    # --------------------------------------------------------

    for use_sex in [
        True,
        False
    ]:

        current_key = (

            model_name,

            use_sex,

            use_resampling,

            use_pca,

            use_umap
        )


        # ----------------------------------------------------
        # SKIP IF ALREADY COMPLETED
        # ----------------------------------------------------

        if current_key in completed_keys:

            print(
                "\nSkipping already completed:"
            )

            print(
                current_key
            )

            continue


        sex_label = (
            "With Sex"
            if use_sex
            else "No Sex"
        )


        print(
            "\n" + "=" * 70
        )

        print(
            f"RUNNING: "
            f"{model_name} | "
            f"SMOTETomek={use_resampling} | "
            f"PCA={use_pca} | "
            f"UMAP={use_umap} | "
            f"{sex_label}"
        )

        print(
            "=" * 70
        )


        # ----------------------------------------------------
        # FEATURES
        # ----------------------------------------------------

        X_features = get_features(
            use_sex
        )


        # ----------------------------------------------------
        # STORAGE
        # ----------------------------------------------------

        y_tests = []

        y_preds = []

        y_preds_proba = []


        # ----------------------------------------------------
        # LOO
        # ----------------------------------------------------

        for train_index, test_index in tqdm(

            loo.split(
                X_features
            ),

            total=loo.get_n_splits(
                X_features
            ),

            desc=(
                f"{model_name} | "
                f"{sex_label} | "
                f"PCA={use_pca} | "
                f"UMAP={use_umap}"
            )
        ):


            # ------------------------------------------------
            # TRAIN / TEST
            # ------------------------------------------------

            X_train = \
                X_features.iloc[
                    train_index
                ].copy()

            X_test = \
                X_features.iloc[
                    test_index
                ].copy()


            y_train = \
                y.iloc[
                    train_index
                ].copy()

            y_test = \
                y.iloc[
                    test_index
                ].copy()


            # ------------------------------------------------
            # 1. SMOTETOMEK
            # ------------------------------------------------

            if use_resampling:

                smote_tomek = SMOTETomek(
                    random_state=42
                )

                X_train, y_train = \
                    smote_tomek.fit_resample(
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

                X_train = \
                    pca.fit_transform(
                        X_train
                    )

                X_test = \
                    pca.transform(
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

                X_train = \
                    umap.fit_transform(
                        X_train,
                        y_train
                    )

                X_test = \
                    umap.transform(
                        X_test
                    )


            # ------------------------------------------------
            # 4. MODEL
            # ------------------------------------------------

            model = clone(
                models[
                    model_name
                ]
            )


            model.fit(
                X_train,
                y_train
            )


            # ------------------------------------------------
            # 5. PREDICTION
            # ------------------------------------------------

            y_pred = model.predict(
                X_test
            )


            y_pred_proba = \
                model.predict_proba(
                    X_test
                )[:, 1]


            # ------------------------------------------------
            # STORE PREDICTIONS
            # ------------------------------------------------

            y_tests.extend(
                y_test.tolist()
            )

            y_preds.extend(
                y_pred.tolist()
            )

            y_preds_proba.extend(
                y_pred_proba.tolist()
            )


        # ====================================================
        # AGGREGATED LOO METRICS
        # ====================================================

        accuracy = accuracy_score(
            y_tests,
            y_preds
        )

        balanced_accuracy = \
            balanced_accuracy_score(
                y_tests,
                y_preds
            )

        recall = recall_score(
            y_tests,
            y_preds,
            zero_division=0
        )

        precision = precision_score(
            y_tests,
            y_preds,
            zero_division=0
        )

        f1 = f1_score(
            y_tests,
            y_preds,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_tests,
            y_preds_proba
        )


        # ====================================================
        # ROC CURVE
        # ====================================================

        fpr, tpr, thresholds = \
            roc_curve(
                y_tests,
                y_preds_proba
            )


        # ====================================================
        # BOOTSTRAP CIs
        # ====================================================

        ci_results = \
            calculate_loo_bootstrap_ci(

                y_true=y_tests,

                y_pred=y_preds,

                y_proba=y_preds_proba,

                n_bootstrap=N_BOOTSTRAP,

                random_state=(
                    BOOTSTRAP_RANDOM_STATE
                )
            )


        # ====================================================
        # COMPLETE RESULT OBJECT
        # ====================================================

        result = {

            # Configuration
            "validation": "LOO",

            "model": model_name,

            "use_sex": use_sex,

            "use_resampling":
                use_resampling,

            "use_pca":
                use_pca,

            "use_umap":
                use_umap,


            # Metrics
            "accuracy":
                accuracy,

            "balanced_accuracy":
                balanced_accuracy,

            "recall":
                recall,

            "precision":
                precision,

            "f1":
                f1,

            "roc_auc":
                roc_auc,


            # CIs
            **ci_results,


            # Number of predictions
            "n_loo_predictions":
                len(y_tests),

            "n_bootstrap":
                N_BOOTSTRAP,

            "CI_level":
                0.95,


            # Complete predictions
            "y_true":
                np.asarray(
                    y_tests
                ),

            "y_pred":
                np.asarray(
                    y_preds
                ),

            "y_proba":
                np.asarray(
                    y_preds_proba
                ),


            # ROC
            "fpr":
                np.asarray(
                    fpr
                ),

            "tpr":
                np.asarray(
                    tpr
                ),

            "thresholds":
                np.asarray(
                    thresholds
                )
        }


        # ====================================================
        # ADD RESULT TO CHECKPOINT
        # ====================================================

        saved_results.append(
            result
        )


        # ====================================================
        # SAVE CHECKPOINT IMMEDIATELY
        #
        # IMPORTANT:
        # This happens after EACH completed analysis.
        # ====================================================

        with open(
            CHECKPOINT_FILE,
            "wb"
        ) as f:

            pickle.dump(
                saved_results,
                f
            )


        # ====================================================
        # UPDATE CSV
        # ====================================================

        csv_rows = []

        for r in saved_results:

            csv_rows.append({

                "validation":
                    r["validation"],

                "model":
                    r["model"],

                "use_sex":
                    r["use_sex"],

                "use_resampling":
                    r["use_resampling"],

                "use_pca":
                    r["use_pca"],

                "use_umap":
                    r["use_umap"],

                "accuracy":
                    r["accuracy"],

                "balanced_accuracy":
                    r["balanced_accuracy"],

                "recall":
                    r["recall"],

                "precision":
                    r["precision"],

                "f1":
                    r["f1"],

                "roc_auc":
                    r["roc_auc"],

                "accuracy_CI_low":
                    r["accuracy_CI_low"],

                "accuracy_CI_high":
                    r["accuracy_CI_high"],

                "balanced_accuracy_CI_low":
                    r["balanced_accuracy_CI_low"],

                "balanced_accuracy_CI_high":
                    r["balanced_accuracy_CI_high"],

                "recall_CI_low":
                    r["recall_CI_low"],

                "recall_CI_high":
                    r["recall_CI_high"],

                "precision_CI_low":
                    r["precision_CI_low"],

                "precision_CI_high":
                    r["precision_CI_high"],

                "f1_CI_low":
                    r["f1_CI_low"],

                "f1_CI_high":
                    r["f1_CI_high"],

                "roc_auc_CI_low":
                    r["roc_auc_CI_low"],

                "roc_auc_CI_high":
                    r["roc_auc_CI_high"],

                "n_loo_predictions":
                    r["n_loo_predictions"],

                "n_bootstrap":
                    r["n_bootstrap"],

                "CI_level":
                    r["CI_level"]
            })


        pd.DataFrame(
            csv_rows
        ).to_csv(
            CSV_FILE,
            index=False
        )


        # ====================================================
        # REPORT COMPLETION
        # ====================================================

        print(
            "\nAnalysis saved successfully."
        )

        print(
            f"ROC AUC = {roc_auc:.3f}"
        )

        print(
            f"Progress: "
            f"{len(saved_results)}/10 analyses saved."
        )


# ============================================================
# FINAL CHECK
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "MACHINE LEARNING ANALYSIS COMPLETE"
)

print(
    "=" * 70
)

print(
    f"Saved analyses: "
    f"{len(saved_results)}/10"
)

print(
    f"\nCheckpoint:"
    f"\n{CHECKPOINT_FILE}"
)

print(
    f"\nCSV:"
    f"\n{CSV_FILE}"
)


# ============================================================
# DISPLAY SUMMARY
# ============================================================

df_summary = pd.DataFrame([

    {

        "validation":
            r["validation"],

        "model":
            r["model"],

        "use_sex":
            r["use_sex"],

        "SMOTETomek":
            r["use_resampling"],

        "PCA":
            r["use_pca"],

        "UMAP":
            r["use_umap"],

        "Accuracy":
            r["accuracy"],

        "Balanced Accuracy":
            r["balanced_accuracy"],

        "Recall":
            r["recall"],

        "Precision":
            r["precision"],

        "F1-score":
            r["f1"],

        "ROC AUC":
            r["roc_auc"]
    }

    for r in saved_results
])


print(
    "\nRESULTS:"
)

print(
    df_summary.to_string(
        index=False
    )
)

"""Part 2 - Figure"""

import pickle
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# LOAD SAVED RESULTS
# ============================================================

with open(
    "LOO_ROC_selected_results.pkl",
    "rb"
) as f:

    saved_results = pickle.load(f)


print(
    f"Loaded {len(saved_results)} saved analyses."
)


# ============================================================
# CHECK
# ============================================================

if len(saved_results) != 10:

    raise ValueError(
        "The checkpoint does not contain "
        "all 10 analyses. "
        "Complete PART 1 first."
    )


# ============================================================
# FIGURE STYLE
# ============================================================

plt.rcParams.update({

    "font.family":
        "sans-serif",

    "font.sans-serif":
        ["DejaVu Sans"],

    "font.size":
        8.5,

    "axes.linewidth":
        1.0,

    "xtick.direction":
        "out",

    "ytick.direction":
        "out"
})


# ============================================================
# FIGURE SIZE
# ============================================================
#
# Width and height can be changed freely.
#
# Current dimensions:
#
#     100 mm wide
#     180 mm high
#
# ============================================================

fig_width_mm = 100
fig_height_mm = 180


fig, axes = plt.subplots(

    2,
    1,

    figsize=(

        fig_width_mm / 25.4,

        fig_height_mm / 25.4
    )
)


# ============================================================
# LINE STYLES
#
# Each configuration has a unique line style.
#
# IMPORTANT:
# The same configuration uses the same line style in
# panels A and B.
#
# This makes the figure completely black-and-white friendly.
# ============================================================

line_style_map = {

    (
        "LogisticRegression",
        True,
        True,
        False
    ):
        "-",

    (
        "LogisticRegression",
        True,
        False,
        False
    ):
        "--",

    (
        "SVC",
        True,
        False,
        False
    ):
        "-.",

    (
        "SVC",
        True,
        False,
        True
    ):
        ":",

    (
        "SVC",
        True,
        True,
        False
    ):
        (0, (5, 2, 1, 2))
}


# ============================================================
# CONFIGURATION LABEL
# ============================================================

def configuration_label(result):

    parts = []

    if result["use_resampling"]:

        parts.append(
            "SMOTETomek"
        )

    if result["use_pca"]:

        parts.append(
            "PCA"
        )

    if result["use_umap"]:

        parts.append(
            "UMAP"
        )

    return " + ".join(parts)


# ============================================================
# MODEL LABEL
# ============================================================

def model_label(result):

    return (

        f'{result["model"]} — '
        f'{configuration_label(result)}'
    )


# ============================================================
# PLOT FUNCTION
# ============================================================

def plot_panel(
    ax,
    use_sex,
    panel_letter
):

    panel_results = [

        r

        for r in saved_results

        if r["use_sex"] == use_sex
    ]


    # --------------------------------------------------------
    # SORT CONFIGURATIONS
    #
    # Keep exactly the order specified in the analysis.
    # --------------------------------------------------------

    config_order = [

        (
            "LogisticRegression",
            True,
            True,
            False
        ),

        (
            "LogisticRegression",
            True,
            False,
            False
        ),

        (
            "SVC",
            True,
            False,
            False
        ),

        (
            "SVC",
            True,
            False,
            True
        ),

        (
            "SVC",
            True,
            True,
            False
        )
    ]


    panel_results = sorted(

        panel_results,

        key=lambda r:
        config_order.index(

            (
                r["model"],

                r["use_resampling"],

                r["use_pca"],

                r["use_umap"]
            )
        )
    )


    # --------------------------------------------------------
    # PLOT ROC CURVES
    #
    # IMPORTANT:
    # No smoothing/interpolation.
    #
    # We plot the original ROC points generated from
    # the aggregated LOO predictions.
    # --------------------------------------------------------

    for r in panel_results:

        fpr = np.asarray(
            r["fpr"]
        )

        tpr = np.asarray(
            r["tpr"]
        )

        auc = r["roc_auc"]


        line_style = line_style_map[

            (
                r["model"],

                r["use_resampling"],

                r["use_pca"],

                r["use_umap"]
            )
        ]


        ax.plot(

            fpr,

            tpr,

            color="black",

            linewidth=1.7,

            linestyle=line_style,

            label=(

                f'{model_label(r)} '
                f'(AUC = {auc:.3f})'
            )
        )


    # --------------------------------------------------------
    # CHANCE LINE
    # --------------------------------------------------------

    ax.plot(

        [0, 1],

        [0, 1],

        color="gray",

        linewidth=0.9,

        linestyle=":"
    )


    # --------------------------------------------------------
    # AXES
    # --------------------------------------------------------

    ax.set_xlim(
        0,
        1
    )

    ax.set_ylim(
        0,
        1
    )

    ax.set_aspect(
        "equal",
        adjustable="box"
    )


    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )


    # --------------------------------------------------------
    # TICKS
    # --------------------------------------------------------

    ax.set_xticks(
        np.linspace(
            0,
            1,
            6
        )
    )

    ax.set_yticks(
        np.linspace(
            0,
            1,
            6
        )
    )


    # --------------------------------------------------------
    # REMOVE TOP / RIGHT SPINES
    # --------------------------------------------------------

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)


    # --------------------------------------------------------
    # PANEL LETTER
    #
    # The A/B is placed at the upper-left corner of the
    # ENTIRE PANEL AREA, including the space occupied by
    # the legend.
    #
    # We use figure coordinates rather than axes coordinates.
    # --------------------------------------------------------

    return panel_results


# ============================================================
# PANEL A — WITH SEX
# ============================================================

panel_A_results = plot_panel(

    axes[0],

    use_sex=True,

    panel_letter="A"
)


# ============================================================
# PANEL B — NO SEX
# ============================================================

panel_B_results = plot_panel(

    axes[1],

    use_sex=False,

    panel_letter="B"
)


# ============================================================
# TITLES
# ============================================================

axes[0].set_title(

    "With Sex",

    fontsize=9,

    loc="left",

    pad=8
)


axes[1].set_title(

    "No Sex",

    fontsize=9,

    loc="left",

    pad=8
)


# ============================================================
# LEGENDS
#
# One legend for EACH PANEL.
#
# The legends are placed below their corresponding ROC plot.
# ============================================================

axes[0].legend(

    loc="upper center",

    bbox_to_anchor=(

        0.5,

        -0.22
    ),

    frameon=False,

    fontsize=6.5,

    ncol=1,

    handlelength=3.0,

    handletextpad=0.7,

    borderaxespad=0
)


axes[1].legend(

    loc="upper center",

    bbox_to_anchor=(

        0.5,

        -0.22
    ),

    frameon=False,

    fontsize=6.5,

    ncol=1,

    handlelength=3.0,

    handletextpad=0.7,

    borderaxespad=0
)


# ============================================================
# PANEL LETTERS
#
# IMPORTANT:
#
# These use FIGURE coordinates.
#
# Therefore A/B refer to the complete panel area,
# including the legend below the ROC graph.
# ============================================================

fig.text(

    0.035,

    0.965,

    "A",

    fontsize=13,

    fontweight="bold",

    ha="left",

    va="top"
)


fig.text(

    0.035,

    0.485,

    "B",

    fontsize=13,

    fontweight="bold",

    ha="left",

    va="top"
)


# ============================================================
# LAYOUT
# ============================================================

plt.subplots_adjust(

    left=0.16,

    right=0.98,

    top=0.94,

    bottom=0.06,

    hspace=0.72
)


# ============================================================
# SAVE TIFF
# ============================================================

plt.savefig(

    "Figure_ROC_LOO_selected.png",

    format="png",

    dpi=600,

    bbox_inches="tight"
)


plt.show()


print(
    "\nFigure saved as:"
)

print(
    "Figure_ROC_LOO_selected.png"
)