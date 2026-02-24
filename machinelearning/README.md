# Machine Learning Analysis

This folder contains scripts for the **supervised machine learning analysis** of gut microbiota data in participants with mood disorders. The analysis explores predictive modeling using microbiota features and sex to distinguish between diagnostic groups.

---

## **Pipeline Overview**

- **Algorithms tested:** 13 supervised machine learning models.  
- **Preprocessing configurations:** 8 combinations using:
  - SMOTETomek for class imbalance correction  
  - PCA for linear dimensionality reduction  
  - UMAP for nonlinear dimensionality reduction  

> All transformations were applied **only to the training set** to avoid data leakage.

- **Validation strategies:**
  - Train-test split  
  - Leave-one-out cross-validation  
  - Stratified K-fold  
  - Repeated stratified K-fold  

- **Predictive features:**  
  - 2,496 ASVs from 16S rRNA sequencing  
  - Sex of the participants (included in some analyses)  

- **Diagnostic comparisons performed:**
  1. Bipolar vs Control (BD vs HC)  
  2. Depression vs Control (MDD vs HC)  
  3. Bipolar vs Depression (BD vs MDD)  
  4. Mood Disorder vs Control (MD vs HC)  

- **Model selection and ranking:**
  - Models were ranked by **F1-score** instead of accuracy to balance precision and recall.  
  - Top 5 performing models were identified.  
  - Subsequent analyses focused on **Mood Disorder vs Control**, with and without sex as a feature.  
  - The two best models were visualized using **ROC curves** to compare performance with and without sex.

---

## **Scripts**

| Script | Description |
|--------|-------------|
| `ML_13models_BDvsHC.py` | Trains and evaluates 13 ML models on Bipolar vs Control data. |
| `ML_13models_BDvsMDD.py` | Trains and evaluates 13 ML models on Bipolar vs Depression data. |
| `ML_13models_MDDvsHC.py` | Trains and evaluates 13 ML models on Depression vs Control data. |
| `ML_13models_MDvsHC.py` | Trains and evaluates 13 ML models on Mood Disorder vs Control data. |
| `ML_MDvsHC_NoSex.py` | Highest-performing models (f1-score) excluding "sex" as a feature. |
| `ROC_curve.py` | Generates ROC curve figure for the best-performing **HistGradientBoosting** and **GaussianNB** models. |

---

## **Dependencies**

These scripts typically require:

```python
# Example Python packages
numpy
pandas
scikit-learn
imblearn
umap-learn
matplotlib
seaborn
