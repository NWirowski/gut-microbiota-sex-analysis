import pandas as pd
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.model_selection import train_test_split
from sklearn.metrics import  (accuracy_score, balanced_accuracy_score, recall_score, precision_score, f1_score, roc_curve, roc_auc_score)
from plotly import graph_objects as go
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
import os

"""HistGradientBoosting with "Sex" as a feature"""

df_sex = pd.read_csv('table.csv')
df_sex['mood'] = df_sex['mood'].map(lambda x: {0: "controle", 2: "TH", 3:"TH"}[x])
df_sex = df_sex.drop(['id'], axis=1)
X_asv_sex = df_sex.drop(['mood', 'sex'], axis=1)
y_sex = df_sex['mood'] == 'TH'
X_asv_relative_sex = (X_asv_sex.T / X_asv_sex.sum(axis=1)).T
X_features_sex = X_asv_relative_sex
X_features_sex['sex'] = df_sex['sex'] == 1

X_trainS, X_testS, y_trainS, y_testS = train_test_split(X_features_sex, y_sex, test_size=0.2, random_state=42)

model_name = "HistGradientBoosting"
hgbS_model = HistGradientBoostingClassifier(random_state=42)

pca = PCA(n_components=0.99, random_state=42)
X_trainS = pca.fit_transform(X_trainS)
X_testS = pca.transform(X_testS)

umap = UMAP(n_components=3, random_state=42)
X_trainS = umap.fit_transform(X_trainS, y_trainS)
X_testS = umap.transform(X_testS)

hgbS_model.fit(X_trainS, y_trainS)

hgbS_y_pred = hgbS_model.predict(X_testS)
hgbS_y_pred

metrics = {
    "validation": "TTS",
    "model": model_name,
    "accuracy": accuracy_score(y_testS, hgbS_y_pred),
    "balanced_accuracy": balanced_accuracy_score(y_testS, hgbS_y_pred),
    "recall": recall_score(y_testS, hgbS_y_pred),
    "precision": precision_score(y_testS, hgbS_y_pred),
    "f1": f1_score(y_testS, hgbS_y_pred),
    "n_features": X_trainS.shape[1],
    "train_size": len(y_trainS),
    "test_size": len(y_testS),
    "timestamp": pd.Timestamp.now(),
}

df_new = pd.DataFrame([metrics])
results_path = "model_results.txt"

if os.path.exists(results_path):
    df_results = pd.read_csv(results_path, sep="\t")
    df_results = pd.concat([df_results, df_new], ignore_index=True)
else:
    df_results = df_new

df_results.to_csv(results_path, sep="\t", index=False)


print("✅ Metrics saved in:", results_path)
print(df_results.tail(3))

hgbS_y_pred_proba = hgbS_model.predict_proba(X_testS)[:,1]
hgbS_y_pred_proba

hgbS_fpr, hgbS_tpr, _ = roc_curve(y_testS, hgbS_y_pred_proba)
hgbS_roc_auc = round(roc_auc_score(y_testS, hgbS_y_pred_proba),2)

"""HistGradientBoosting without "Sex" as a feature"""

df = pd.read_csv('table.csv')
df['mood'] = df['mood'].map(lambda x: {0: "controle", 2: "TH", 3:"TH"}[x])
df = df.drop(['id', 'sex'], axis=1)
X_asv = df.drop(['mood'], axis=1)
y = df['mood'] == 'TH'
X_asv_relative = (X_asv.T / X_asv.sum(axis=1)).T
X_features = X_asv_relative

X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

model_name = "HistGradientBoosting_NoSex"
hgb_model = HistGradientBoostingClassifier(random_state=42)

pca = PCA(n_components=0.99, random_state=42)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

umap = UMAP(n_components=3, random_state=42)
X_train = umap.fit_transform(X_train, y_train)
X_test = umap.transform(X_test)

hgb_model.fit(X_train, y_train)

hgb_y_pred = hgb_model.predict(X_test)
hgb_y_pred

metrics = {
    "validation": "TTS",
    "model": model_name,
    "accuracy": accuracy_score(y_test, hgb_y_pred),
    "balanced_accuracy": balanced_accuracy_score(y_test, hgb_y_pred),
    "recall": recall_score(y_test, hgb_y_pred),
    "precision": precision_score(y_test, hgb_y_pred),
    "f1": f1_score(y_test, hgb_y_pred),
    "n_features": X_train.shape[1],
    "train_size": len(y_train),
    "test_size": len(y_test),
    "timestamp": pd.Timestamp.now(),
}

df_new = pd.DataFrame([metrics])
results_path = "model_results.txt"

if os.path.exists(results_path):
    df_results = pd.read_csv(results_path, sep="\t")
    df_results = pd.concat([df_results, df_new], ignore_index=True)
else:
    df_results = df_new

df_results.to_csv(results_path, sep="\t", index=False)


print("✅ Metrics saved in:", results_path)
print(df_results.tail(3))



hgb_y_pred_proba = hgb_model.predict_proba(X_test)[:,1]
hgb_y_pred_proba

hgb_fpr, hgb_tpr, _ = roc_curve(y_test, hgb_y_pred_proba)
hgb_roc_auc = round(roc_auc_score(y_test, hgb_y_pred_proba),2)

from plotly import graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(x=hgbS_fpr, y=hgbS_tpr, mode='lines', name=f'HistGradientBoosting with "Sex" as a feature (ROC AUC: {hgbS_roc_auc})'))

fig.add_trace(go.Scatter(x=hgb_fpr, y=hgb_tpr, mode='lines', name=f'HistGradientBoosting without "Sex" as a feature (ROC AUC: {hgb_roc_auc})'))

fig.add_shape(type='line', line=dict(dash='dash'),
              x0=0, x1=1, y0=0, y1=1)

fig.update_layout(title='ROC Curve Comparison - Histogram-based Gradient Boosting Classification Tree',
                  xaxis_title='False Positive Rate',
                  yaxis_title='True Positive Rate',
                  legend=dict(x=0.7, y=0.05))

fig.show()

import numpy as np
import plotly.graph_objects as go

fpr_smooth = np.linspace(0, 1, 1000)

tpr_smooth_S = np.interp(fpr_smooth, hgbS_fpr, hgbS_tpr)
tpr_smooth_noS = np.interp(fpr_smooth, hgb_fpr, hgb_tpr)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=fpr_smooth, y=tpr_smooth_S,
    mode='lines', line=dict(width=3),
    name=f'HistGradientBoosting with "Sex" (ROC AUC: {hgbS_roc_auc})'
))

fig.add_trace(go.Scatter(
    x=fpr_smooth, y=tpr_smooth_noS,
    mode='lines', line=dict(width=3),
    name=f'HistGradientBoosting without "Sex" (ROC AUC: {hgb_roc_auc})'
))

fig.add_shape(type='line', line=dict(dash='dash'),
              x0=0, x1=1, y0=0, y1=1)

fig.update_layout(
    title='ROC Curve Comparison - Histogram-based Gradient Boosting Classification Tree',
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate',
    legend=dict(x=0.6, y=0.05),
    width=800, height=600
)

fig.show()

import numpy as np
print("With sex (HGB):", len(np.unique(hgbS_y_pred_proba)))
print("Without sex (HGB):", len(np.unique(hgb_y_pred_proba)))