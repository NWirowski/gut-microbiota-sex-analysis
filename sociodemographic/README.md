# Sociodemographic Analysis

This folder contains the R Markdown analysis of sociodemographic and clinical characteristics of the study participants using the revised diagnostic grouping.

## Study Groups

Participants are classified into two groups:

* **Control** — participants without a mood disorder
* **Mood Disorder** — participants diagnosed with Bipolar Disorder (BD) or Major Depressive Disorder (MDD)

The original three-group analyses (BD, MDD, and Control) are retained separately in [`/original_group_analyses`](../original_group_analyses/).

## Data

The analysis uses the study metadata file (`metadata.txt`), which contains sociodemographic, clinical, and microbiota-related covariates.

### Categorical variables

| Variable      | Description                                                    | Coding                           |
| ------------- | -------------------------------------------------------------- | -------------------------------- |
| `thdico`      | Group                                                          | 0 = Control; 1 = Mood Disorder   |
| `corpele`     | Skin Color                                                     | 1 = White; 2 = Non-White         |
| `sexo`        | Sex                                                            | 1 = Male; 2 = Female             |
| `ABEP`        | Socioeconomic status                                           | 1 = Upper class; 2 = Lower class |
| `histpsifam`  | Family history of psychiatric disorder                         | 1 = Yes; 0 = No                  |
| `a20medicpsi` | Use of psychiatric medication                                  | 1 = Yes; 0 = No                  |
| `antib`       | Antibiotic use in the last month                               | 1 = Yes; 0 = No                  |
| `suplem`      | Use of prebiotics, probiotics, or synbiotics in the last month | 1 = Yes; 0 = No                  |

### Quantitative variables

| Variable     | Description           |
| ------------ | --------------------- |
| `idade`      | Age                   |
| `anosestudo` | Years of schooling    |
| `IMC`        | Body Mass Index (BMI) |

Categorical variables are converted to factors with descriptive labels before analysis, while quantitative variables are converted to numeric format.

## Analyses

### Descriptive statistics

Descriptive statistics are calculated for quantitative variables, including:

* Mean
* Standard deviation
* Median
* Interquartile range (IQR)

Statistics are calculated both overall and separately by study group.

### Distribution assessment

Histograms are generated for:

* Age
* Years of schooling
* BMI

Gaussian curves are overlaid on the distributions to assist with assessment of the data distribution.

### Quantitative variables

Group comparisons are performed using both parametric and non-parametric approaches.

**Parametric analysis:**

* Independent-samples t-test

**Non-parametric analysis:**

* Wilcoxon rank-sum test (Mann–Whitney U test)

These tests are applied to:

* Age
* Years of schooling
* BMI

### Categorical variables

Absolute and relative frequencies are calculated for categorical variables overall and by study group.

The following variables are evaluated:

* Skin color
* Sex
* Socioeconomic status
* Family history of psychiatric disorder
* Psychiatric medication use
* Antibiotic use
* Prebiotic/probiotic/synbiotic use

Group comparisons use:

* Pearson's chi-square test
* Chi-square test without continuity correction
* Fisher's exact test

Expected frequencies are also examined to assess the appropriateness of the chi-square test.

## Software and Packages

The analysis is performed in **R** using R Markdown.

The main packages used include:

* `readr` — importing metadata
* `tidyverse` — data manipulation and descriptive statistics
* `janitor` — frequency tables and tabulation

## Reproducibility

The R Markdown file in this folder contains the complete analysis workflow, from metadata import and variable labeling through descriptive statistics and statistical comparisons between Mood Disorder and Control participants.

The metadata path in the original analysis is configured for a local data directory and may need to be adjusted according to the user's local file structure.
