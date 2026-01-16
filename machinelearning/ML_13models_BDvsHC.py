from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.decomposition import PCA
from sklearn.metrics import (balanced_accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, accuracy_score)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, BaggingClassifier, AdaBoostClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.combine import SMOTETomek
from umap import UMAP
from tqdm.notebook import tqdm
from plotly import express as ex
import pandas as pd
import warnings

df = pd.read_csv('table.csv')
df['mood'] = df['mood'].map(lambda x: {0: "controle", 2: "bipolar", 3:"depressão"}[x])
df = df[df['mood'].isin(["bipolar", "controle"])]

df = df.drop(['id', 'sex'], axis=1)
X_asv = df.drop(['mood'], axis=1)
y = df['mood'] == 'bipolar'
X_asv_relative = (X_asv.T / X_asv.sum(axis=1)).T
X_features = X_asv_relative

"""**Train test split**"""

from umap import umap_

# filtra apenas warnings do UMAP que contêm essa mensagem específica
warnings.filterwarnings(
    "ignore",
    message="n_jobs value 1 overridden to 1 by setting random_state.*",
    category=UserWarning
)

# -----------------------------
# Definição dos modelos
# -----------------------------
base_tree = DecisionTreeClassifier(max_depth=5, random_state=42)

models = {
    "LogisticRegression": LogisticRegression(max_iter=200, random_state=42),
    "SVC": SVC(kernel="rbf", probability=True, random_state=42),
    "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000, random_state=42),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
    "GaussianNB": GaussianNB(),
    "DecisionTree": DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42),
    "Bagging": BaggingClassifier(estimator=base_tree, n_estimators=50, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance"),
    "CatBoost": CatBoostClassifier(verbose=False, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=500, max_depth=None,  min_samples_leaf=2, class_weight="balanced", random_state=42),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=500, max_depth=None, min_samples_split=2, min_samples_leaf=2, class_weight="balanced", random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, min_samples_leaf=2, random_state=42),
}

# -----------------------------
# Configurações SMOTE/PCA/UMAP
# -----------------------------
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

results = []

# -----------------------------
# Loop principal
# -----------------------------
for model_name, model in models.items():
    for use_resampling, use_pca, use_umap in configs:
        # Split inicial
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y, test_size=0.2, random_state=42
        )

        # 1. Resampling (apenas treino)
        if use_resampling:
            smote_tomek = SMOTETomek(random_state=42)
            X_train, y_train = smote_tomek.fit_resample(X_train, y_train)

        # 2. PCA
        if use_pca:
            pca = PCA(n_components=0.99, random_state=42)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

        # 3. UMAP
        if use_umap:
            umap = UMAP(n_components=3, random_state=42)
            X_train = umap.fit_transform(X_train, y_train)
            X_test = umap.transform(X_test)

        # Treino do modelo
        model.fit(X_train, y_train)

        # Predições
        y_preds = model.predict(X_test)
        y_preds_proba = None
        try:
            y_preds_proba = model.predict_proba(X_test)[:, 1]
        except Exception:
            pass  # alguns modelos podem não ter predict_proba

        # Métricas
        metrics = {
            "validation": "TTS",
            "model": model_name,
            "use_resampling": use_resampling,
            "use_pca": use_pca,
            "use_umap": use_umap,
            "accuracy": accuracy_score(y_test, y_preds),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_preds),
            "recall": recall_score(y_test, y_preds),
            "precision": precision_score(y_test, y_preds),
            "f1": f1_score(y_test, y_preds),
        }

        if y_preds_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y_test, y_preds_proba)
        else:
            metrics["roc_auc"] = None

        results.append(metrics)

# DataFrame final com resultados
df_results = pd.DataFrame(results)
print(df_results.head())

"""**Leave One Out**"""

from sklearn.model_selection import LeaveOneOut
import numpy as np
from tqdm import tqdm  # para barra de progresso

# -----------------------------
# Loop LOO
# -----------------------------
loo = LeaveOneOut()

for model_name, model in models.items():
    for use_resampling, use_pca, use_umap in configs:

        y_tests = []
        y_preds = []
        y_preds_proba = []

        for train_index, test_index in tqdm(loo.split(X_features), total=loo.get_n_splits(X_features),
                                           desc=f"{model_name} LOO, PCA={use_pca}, UMAP={use_umap}, SMOTE={use_resampling}"):
            X_train, X_test = X_features.iloc[train_index], X_features.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # 1. Resampling (apenas treino)
            if use_resampling:
                smote_tomek = SMOTETomek(random_state=42)
                X_train, y_train = smote_tomek.fit_resample(X_train, y_train)

            # 2. PCA
            if use_pca:
                pca = PCA(n_components=0.99, random_state=42)
                X_train = pca.fit_transform(X_train)
                X_test = pca.transform(X_test)

            # 3. UMAP
            if use_umap:
                umap = UMAP(n_components=3, random_state=42)
                X_train = umap.fit_transform(X_train, y_train)
                X_test = umap.transform(X_test)

            # Treino do modelo
            model.fit(X_train, y_train)

            # Predições
            y_pred = model.predict(X_test)
            y_tests.extend(y_test)
            y_preds.extend(y_pred)

            # Probabilidades para ROC AUC, se disponíveis
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                y_preds_proba.extend(y_pred_proba)
            except Exception:
                y_preds_proba = None

        # Métricas
        metrics = {
            "validation": "LOO",
            "model": model_name,
            "use_resampling": use_resampling,
            "use_pca": use_pca,
            "use_umap": use_umap,
            "accuracy": accuracy_score(y_tests, y_preds),
            "balanced_accuracy": balanced_accuracy_score(y_tests, y_preds),
            "recall": recall_score(y_tests, y_preds),
            "precision": precision_score(y_tests, y_preds),
            "f1": f1_score(y_tests, y_preds),
        }

        if y_preds_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y_tests, y_preds_proba)
        else:
            metrics["roc_auc"] = None

        results.append(metrics)  # adiciona ao mesmo results, sem sobrescrever TTS

# Atualiza DataFrame final
df_results = pd.DataFrame(results)
print(df_results.head())

"""**Stratified K Fold**

"""

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from tqdm import tqdm
import pandas as pd

# -----------------------------
# Configurações do StratifiedKFold
# -----------------------------
N_SPLITS = 10
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# -----------------------------
# Loop StratifiedKFold
# -----------------------------
for model_name, model in models.items():
    for use_resampling, use_pca, use_umap in configs:

        y_tests = []
        y_preds = []
        y_preds_proba = []

        for train_index, test_index in tqdm(
            skf.split(X_features, y),
            total=skf.get_n_splits(X_features, y),
            desc=f"{model_name} SKF, PCA={use_pca}, UMAP={use_umap}, SMOTE={use_resampling}"
        ):
            X_train, X_test = X_features.iloc[train_index], X_features.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # 1. Resampling (apenas treino)
            if use_resampling:
                smote_tomek = SMOTETomek(random_state=42)
                X_train, y_train = smote_tomek.fit_resample(X_train, y_train)

            # 2. PCA
            if use_pca:
                pca = PCA(n_components=0.99, random_state=42)
                X_train = pca.fit_transform(X_train)
                X_test = pca.transform(X_test)

            # 3. UMAP
            if use_umap:
                umap = UMAP(n_components=3, random_state=42)
                X_train = umap.fit_transform(X_train, y_train)
                X_test = umap.transform(X_test)

            # Treino do modelo
            model.fit(X_train, y_train)

            # Predições
            y_pred = model.predict(X_test)
            y_tests.extend(y_test)
            y_preds.extend(y_pred)

            # Probabilidades para ROC AUC, se disponíveis
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                y_preds_proba.extend(y_pred_proba)
            except Exception:
                y_preds_proba = None

        # Métricas
        metrics = {
            "validation": "StratifiedKFold",
            "model": model_name,
            "use_resampling": use_resampling,
            "use_pca": use_pca,
            "use_umap": use_umap,
            "accuracy": accuracy_score(y_tests, y_preds),
            "balanced_accuracy": balanced_accuracy_score(y_tests, y_preds),
            "recall": recall_score(y_tests, y_preds),
            "precision": precision_score(y_tests, y_preds),
            "f1": f1_score(y_tests, y_preds),
        }

        if y_preds_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y_tests, y_preds_proba)
        else:
            metrics["roc_auc"] = None

        results.append(metrics)  # adiciona aos resultados existentes, sem sobrescrever

# Atualiza DataFrame final
df_results = pd.DataFrame(results)
print(df_results.tail())  # mostra os últimos resultados, que são do SKF

"""**Repeated Stratified K Fold**"""

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from tqdm import tqdm
import pandas as pd

# -----------------------------
# Configurações do RepeatedStratifiedKFold
# -----------------------------
N_SPLITS = 10
N_REPEATS = 3
rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=42)

# -----------------------------
# Loop RepeatedStratifiedKFold
# -----------------------------
for model_name, model in models.items():
    for use_resampling, use_pca, use_umap in configs:

        y_tests = []
        y_preds = []
        y_preds_proba = []

        for train_index, test_index in tqdm(
            rskf.split(X_features, y),
            total=rskf.get_n_splits(X_features, y),
            desc=f"{model_name} RSKF, PCA={use_pca}, UMAP={use_umap}, SMOTE={use_resampling}"
        ):
            X_train, X_test = X_features.iloc[train_index], X_features.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # 1. Resampling (apenas treino)
            if use_resampling:
                smote_tomek = SMOTETomek(random_state=42)
                X_train, y_train = smote_tomek.fit_resample(X_train, y_train)

            # 2. PCA
            if use_pca:
                pca = PCA(n_components=0.99, random_state=42)
                X_train = pca.fit_transform(X_train)
                X_test = pca.transform(X_test)

            # 3. UMAP
            if use_umap:
                umap = UMAP(n_components=3, random_state=42)
                X_train = umap.fit_transform(X_train, y_train)
                X_test = umap.transform(X_test)

            # Treino do modelo
            model.fit(X_train, y_train)

            # Predições
            y_pred = model.predict(X_test)
            y_tests.extend(y_test)
            y_preds.extend(y_pred)

            # Probabilidades para ROC AUC, se disponíveis
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                y_preds_proba.extend(y_pred_proba)
            except Exception:
                y_preds_proba = None

        # Métricas
        metrics = {
            "validation": "RepeatedStratifiedKFold",
            "model": model_name,
            "use_resampling": use_resampling,
            "use_pca": use_pca,
            "use_umap": use_umap,
            "accuracy": accuracy_score(y_tests, y_preds),
            "balanced_accuracy": balanced_accuracy_score(y_tests, y_preds),
            "recall": recall_score(y_tests, y_preds),
            "precision": precision_score(y_tests, y_preds),
            "f1": f1_score(y_tests, y_preds),
        }

        if y_preds_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y_tests, y_preds_proba)
        else:
            metrics["roc_auc"] = None

        results.append(metrics)  # adiciona aos resultados existentes, sem sobrescrever

# Atualiza DataFrame final
df_results = pd.DataFrame(results)
print(df_results.tail())  # mostra os últimos resultados, que são do RSKF

df_results.to_csv("df_results.csv", index=False)

"""MLP que não convergiu"""

# from sklearn.model_selection import train_test_split, LeaveOneOut, StratifiedKFold, RepeatedStratifiedKFold
# from sklearn.decomposition import PCA
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
# from imblearn.combine import SMOTETomek
# from umap import UMAP
# from tqdm import tqdm
# import pandas as pd

# # -----------------------------
# # Configurações
# # -----------------------------
# results_mlp = []
# configs = [
#     (False, False, False),
#     (False, False, True),
#     (True,  False, False),
#     (True,  False, True),
#     (False, True,  False),
#     (False, True,  True),
#     (True,  True,  False),
#     (True,  True,  True),
# ]

# # Modelo MLP com max_iter maior
# mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=5000, random_state=42)

# # -----------------------------
# # Função de avaliação
# # -----------------------------
# def evaluate_and_store(validation_name, y_tests, y_preds, y_preds_proba, use_resampling, use_pca, use_umap):
#     metrics = {
#         "validation": validation_name,
#         "model": "MLP",
#         "use_resampling": use_resampling,
#         "use_pca": use_pca,
#         "use_umap": use_umap,
#         "accuracy": accuracy_score(y_tests, y_preds),
#         "balanced_accuracy": balanced_accuracy_score(y_tests, y_preds),
#         "recall": recall_score(y_tests, y_preds),
#         "precision": precision_score(y_tests, y_preds),
#         "f1": f1_score(y_tests, y_preds),
#     }
#     if y_preds_proba is not None and len(y_preds_proba) == len(y_tests):
#         try:
#             metrics["roc_auc"] = roc_auc_score(y_tests, y_preds_proba)
#         except Exception:
#             metrics["roc_auc"] = None
#     else:
#         metrics["roc_auc"] = None
#     results_mlp.append(metrics)

# # -----------------------------
# # 1. Train/Test Split
# # -----------------------------
# for use_resampling, use_pca, use_umap in configs:
#     X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

#     # Resampling
#     if use_resampling:
#         smote_tomek = SMOTETomek(random_state=42)
#         X_train, y_train = smote_tomek.fit_resample(X_train, y_train)

#     # PCA
#     if use_pca:
#         pca = PCA(n_components=0.99, random_state=42)
#         X_train = pca.fit_transform(X_train)
#         X_test = pca.transform(X_test)

#     # UMAP
#     if use_umap:
#         umap = UMAP(n_components=3, random_state=42)
#         X_train = umap.fit_transform(X_train, y_train)
#         X_test = umap.transform(X_test)

#     # Modelo
#     mlp.fit(X_train, y_train)
#     y_preds = mlp.predict(X_test)
#     try:
#         y_preds_proba = mlp.predict_proba(X_test)[:, 1]
#     except:
#         y_preds_proba = None

#     evaluate_and_store("TTS", y_test, y_preds, y_preds_proba, use_resampling, use_pca, use_umap)

# # -----------------------------
# # 2. Leave-One-Out
# # -----------------------------
# loo = LeaveOneOut()
# for use_resampling, use_pca, use_umap in configs:
#     y_tests, y_preds, y_preds_proba = [], [], []
#     for train_index, test_index in tqdm(loo.split(X_features), total=loo.get_n_splits(X_features),
#                                         desc=f"MLP LOO, PCA={use_pca}, UMAP={use_umap}, SMOTE={use_resampling}"):
#         X_train, X_test = X_features.iloc[train_index], X_features.iloc[test_index]
#         y_train, y_test = y.iloc[train_index], y.iloc[test_index]

#         if use_resampling:
#             smote_tomek = SMOTETomek(random_state=42)
#             X_train, y_train = smote_tomek.fit_resample(X_train, y_train)
#         if use_pca:
#             pca = PCA(n_components=0.99, random_state=42)
#             X_train = pca.fit_transform(X_train)
#             X_test = pca.transform(X_test)
#         if use_umap:
#             umap = UMAP(n_components=3, random_state=42)
#             X_train = umap.fit_transform(X_train, y_train)
#             X_test = umap.transform(X_test)

#         mlp.fit(X_train, y_train)
#         y_pred = mlp.predict(X_test)
#         y_tests.extend(y_test)
#         y_preds.extend(y_pred)
#         try:
#             y_pred_proba = mlp.predict_proba(X_test)[:, 1]
#             y_preds_proba.extend(y_pred_proba)
#         except:
#             y_preds_proba = None
#     evaluate_and_store("LOO", y_tests, y_preds, y_preds_proba, use_resampling, use_pca, use_umap)

# # -----------------------------
# # 3. StratifiedKFold
# # -----------------------------
# skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# for use_resampling, use_pca, use_umap in configs:
#     y_tests, y_preds, y_preds_proba = [], [], []
#     for train_index, test_index in tqdm(skf.split(X_features, y), total=skf.get_n_splits(X_features),
#                                         desc=f"MLP SKF, PCA={use_pca}, UMAP={use_umap}, SMOTE={use_resampling}"):
#         X_train, X_test = X_features.iloc[train_index], X_features.iloc[test_index]
#         y_train, y_test = y.iloc[train_index], y.iloc[test_index]

#         if use_resampling:
#             smote_tomek = SMOTETomek(random_state=42)
#             X_train, y_train = smote_tomek.fit_resample(X_train, y_train)
#         if use_pca:
#             pca = PCA(n_components=0.99, random_state=42)
#             X_train = pca.fit_transform(X_train)
#             X_test = pca.transform(X_test)
#         if use_umap:
#             umap = UMAP(n_components=3, random_state=42)
#             X_train = umap.fit_transform(X_train, y_train)
#             X_test = umap.transform(X_test)

#         mlp.fit(X_train, y_train)
#         y_pred = mlp.predict(X_test)
#         y_tests.extend(y_test)
#         y_preds.extend(y_pred)
#         try:
#             y_pred_proba = mlp.predict_proba(X_test)[:, 1]
#             y_preds_proba.extend(y_pred_proba)
#         except:
#             y_preds_proba = None
#     evaluate_and_store("StratifiedKFold", y_tests, y_preds, y_preds_proba, use_resampling, use_pca, use_umap)

# # -----------------------------
# # 4. RepeatedStratifiedKFold
# # -----------------------------
# rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
# for use_resampling, use_pca, use_umap in configs:
#     y_tests, y_preds, y_preds_proba = [], [], []
#     for train_index, test_index in tqdm(rskf.split(X_features, y), total=rskf.get_n_splits(X_features, y),
#                                         desc=f"MLP RSKF, PCA={use_pca}, UMAP={use_umap}, SMOTE={use_resampling}"):
#         X_train, X_test = X_features.iloc[train_index], X_features.iloc[test_index]
#         y_train, y_test = y.iloc[train_index], y.iloc[test_index]

#         if use_resampling:
#             smote_tomek = SMOTETomek(random_state=42)
#             X_train, y_train = smote_tomek.fit_resample(X_train, y_train)
#         if use_pca:
#             pca = PCA(n_components=0.99, random_state=42)
#             X_train = pca.fit_transform(X_train)
#             X_test = pca.transform(X_test)
#         if use_umap:
#             umap = UMAP(n_components=3, random_state=42)
#             X_train = umap.fit_transform(X_train, y_train)
#             X_test = umap.transform(X_test)

#         mlp.fit(X_train, y_train)
#         y_pred = mlp.predict(X_test)
#         y_tests.extend(y_test)
#         y_preds.extend(y_pred)
#         try:
#             y_pred_proba = mlp.predict_proba(X_test)[:, 1]
#             y_preds_proba.extend(y_pred_proba)
#         except:
#             y_preds_proba = None
#     evaluate_and_store("RepeatedStratifiedKFold", y_tests, y_preds, y_preds_proba, use_resampling, use_pca, use_umap)

# # -----------------------------
# # DataFrame final com todos os resultados
# # -----------------------------
# df_results_mlp = pd.DataFrame(results_mlp)
# print(df_results_mlp.tail())

# df_results_mlp.to_csv("df_results_mlp.csv", index=False)

"""Salvar progresso para outro dia"""

# import pickle

# all_vars = {
#     "df": df,
#     "X_asv": X_asv,
#     "y": y,
#     "X_asv_relative": X_asv_relative,
#     "X_features": X_features,
#     "models": models,
#     "configs": configs,
#     "results": results,
#     "df_results": df_results
# }

# with open("all_vars.pkl", "wb") as f:
#     pickle.dump(all_vars, f)

# print("Todas as variáveis foram salvas em 'all_vars.pkl'.")

# from google.colab import files
# files.download("all_vars.pkl")

"""Carregar progresso outro dia"""

# import pickle

# filename = "all_vars.pkl"

# with open(filename, "rb") as f:
#     all_vars = pickle.load(f)

# df = all_vars["df"]
# X_asv = all_vars["X_asv"]
# y = all_vars["y"]
# X_asv_relative = all_vars["X_asv_relative"]
# X_features = all_vars["X_features"]
# models = all_vars["models"]
# configs = all_vars["configs"]
# results = all_vars["results"]
# df_results = all_vars["df_results"]

# print("Todas as variáveis foram carregadas com sucesso!")