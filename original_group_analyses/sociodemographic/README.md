# Sociodemographic Analysis

This folder contains the R Markdown notebook for **sociodemographic and clinical data analysis** from the gut microbiota study comparing Bipolar, Depression, and Control groups.  

---

## Notebook

**File:** `Sociodemographic Analysis.Rmd`  

**Author:** Natália Wirowski  
**Date:** 2025-10-16  

**Purpose:**  
- Import and clean metadata from participants  
- Label categorical variables with meaningful levels  
- Assess quantitative variables (age and years of schooling)  
- Perform descriptive statistics and visualizations  
- Conduct statistical tests to compare groups  

---

## **Variables Analyzed**

### Qualitative (categorical) variables

| Variable | Label | Levels |
|----------|-------|--------|
| `mood` | Group | 0 = Control, 2 = Bipolar, 3 = Depression |
| `corpele` | Skin color | 1 = White, 2 = Non-White |
| `sexo` | Sex | 1 = Male, 2 = Female |
| `ABEP` | Socioeconomic status | 1 = Upper class, 2 = Lower class |
| `histpsifam` | Familial psychiatric history | 1 = Yes, 0 = No |

### Quantitative variables

| Variable | Description |
|----------|------------|
| `idade` | Age |
| `anosestudo` | Years of schooling |

---

## **Analyses Performed**

1. **Labeling and factor conversion** – ensuring categorical variables have correct levels.  
2. **Descriptive statistics** – mean, median, standard deviation, interquartile range.  
3. **Visualizations** – histograms for quantitative variables.  
4. **Parametric tests** – ANOVA for age and years of schooling.  
5. **Non-parametric tests** – Kruskal-Wallis tests for age and schooling.  
6. **Chi-square / Fisher tests** – for categorical variables (sex, skin color) by group.  

---

## **Dependencies**

The notebook uses the following R packages:

```r
library(readr)
library(tidyverse)
