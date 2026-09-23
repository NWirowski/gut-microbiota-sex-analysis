# Microbiota Analysis

This folder contains the R scripts used for the revised microbiota analyses comparing **Mood Disorder** participants with **Healthy Controls**.

The revised diagnostic grouping combines participants with Bipolar Disorder (BD) or Major Depressive Disorder (MDD) into a single **Mood Disorder** group. The original three-group analyses (BD, MDD, and Control) are retained separately in [`/original_group_analyses`](../original_group_analyses/).

## Study Groups

The primary grouping variable is:

| Group             | Definition                                                                |
| ----------------- | ------------------------------------------------------------------------- |
| **Control**       | Participants without a mood disorder                                      |
| **Mood Disorder** | Participants diagnosed with Bipolar Disorder or Major Depressive Disorder |

Sex-stratified analyses additionally evaluate Mood Disorder vs Control separately in males and females, while interaction analyses evaluate whether the association between diagnosis and microbiota differs by sex.

## Scripts

### `mood_disorder_vs_control.R`

Main microbiota analysis comparing Mood Disorder and Control participants.

The script includes:

#### Alpha diversity

Alpha diversity is estimated using:

* Observed ASVs
* Shannon diversity
* Inverse Simpson diversity

Group differences are evaluated using the **Wilcoxon rank-sum test (Mann–Whitney test)**.

Boxplots with individual observations are generated for each diversity metric.

#### Beta diversity

Beta diversity is assessed using **Bray–Curtis dissimilarity**.

The analysis includes:

* Bray–Curtis distance calculation
* Principal Coordinates Analysis (PCoA)
* PERMANOVA with 999 permutations

PCoA plots display the two study groups and the percentage of variation explained by the first two principal coordinates.

#### Sex × diagnosis interaction

The script also evaluates potential sex × diagnosis effects on alpha and beta diversity.

For alpha diversity, factorial ANOVA models are fitted:

```text
Diversity metric ~ Sex * Diagnosis
```

For beta diversity, PERMANOVA is performed using:

```text
Bray-Curtis distance ~ Sex * Diagnosis
```

Multivariate dispersion is assessed using `betadisper`.

#### Sex-stratified diversity analyses

Alpha and beta diversity are additionally analyzed separately in:

* Females: Mood Disorder vs Control
* Males: Mood Disorder vs Control

For each sex, alpha diversity is compared using the Wilcoxon rank-sum test and beta diversity is evaluated using Bray–Curtis dissimilarity, PCoA, and PERMANOVA.

### `differential_abundance_analysis.R`

Performs the primary **differential abundance analysis** comparing Mood Disorder and Control participants.

Differential abundance is assessed using **DESeq2**.

The adjusted model includes:

```text
ABEP + psychiatric medication use + BMI + antibiotic use +
prebiotic/probiotic/synbiotic use + Mood Disorder
```

BMI is centered and scaled before inclusion in the model.

The following covariates are included as categorical variables:

* Socioeconomic status (`ABEP`)
* Psychiatric medication use (`a20medicpsi`)
* Antibiotic use in the previous month (`antib`)
* Prebiotic/probiotic/synbiotic use in the previous month (`suplem`)

Samples with missing values in the required covariates are excluded from the adjusted model.

A pseudocount is added to the ASV count table before DESeq2 analysis.

Results are classified according to the direction of the log2 fold change:

* Positive log2 fold change — higher abundance in the Mood Disorder group
* Negative log2 fold change — lower abundance in the Mood Disorder group

Results are filtered using:

```text
adjusted p-value < 0.05
```

Taxonomic information is added to the differential abundance results, including:

* Kingdom
* Phylum
* Class
* Order
* Family
* Genus
* Species

Full results and the top five ASVs in each direction are exported as tab-separated text files.

### `differential_abundance_analysis_sex.R`

Performs **sex-stratified differential abundance analyses**.

Separate phyloseq objects are created for:

* Female participants
* Male participants

Within each sex, Mood Disorder participants are compared with Controls using the same adjusted DESeq2 model as the primary differential abundance analysis:

```text
ABEP + psychiatric medication use + BMI + antibiotic use +
prebiotic/probiotic/synbiotic use + Mood Disorder
```

Differentially abundant ASVs are identified using:

```text
adjusted p-value < 0.05
```

Results are separated according to the direction of the log2 fold change and exported separately for females and males.

Top-five ASV tables are also generated for each sex and direction of association.

These sex-stratified analyses are intended to explore whether differential abundance patterns are consistent across males and females.

### `differential_abundance_analysis_interaction.R`

Evaluates **sex × Mood Disorder interaction effects** on ASV abundance.

The DESeq2 model is:

```text
Sex + Mood Disorder + Sex × Mood Disorder
```

The interaction term evaluates whether the association between Mood Disorder status and ASV abundance differs according to sex.

Interaction results are filtered using:

```text
adjusted p-value < 0.05
```

Significant interaction effects are separated according to the direction of the log2 fold change and annotated with taxonomic information.

The script exports:

* Positive interaction effects
* Negative interaction effects
* Top five positive interaction effects
* Top five negative interaction effects

## Statistical Threshold

For differential abundance analyses, statistical significance is defined as:

```text
adjusted p-value < 0.05
```

The DESeq2 results are ordered according to the adjusted p-value.

## Main R Packages

The analyses use the following R packages:

* `phyloseq` — microbiome data management and diversity analysis
* `DESeq2` — differential abundance analysis
* `vegan` — PERMANOVA, multivariate dispersion, and ecological distance analyses
* `dada2` — microbiome sequence-processing workflow and diversity-related analyses
* `ggplot2` — data visualization
* `dplyr` / `tidyverse` — data manipulation
* `haven` — handling labelled metadata variables
* `Biostrings` — biological sequence manipulation
* `patchwork` — combining plots

## Input Data

The scripts use processed microbiome data stored as **phyloseq objects**.

The main objects used are:

* `phyloseq_object_adjusted.rds` — phyloseq object used for adjusted differential abundance analyses
* `phyloseq_object_complete.rds` — phyloseq object used for diversity analyses

Taxonomic annotation for ASVs is obtained from the ASV taxonomy table.

## Output

The scripts generate tab-separated text files containing:

* Alpha diversity results
* Alpha diversity p-values
* PERMANOVA results
* Differential abundance results
* Sex-stratified differential abundance results
* Sex × diagnosis interaction results
* Top five differentially abundant ASVs and interaction effects

The scripts also generate graphical outputs including:

* Alpha diversity boxplots
* Bray–Curtis PCoA plots
* Sex-stratified PCoA plots

## Analysis Overview

| Analysis                              | Grouping                            | Method                         |
| ------------------------------------- | ----------------------------------- | ------------------------------ |
| Alpha diversity                       | Mood Disorder vs Control            | Wilcoxon rank-sum test         |
| Beta diversity                        | Mood Disorder vs Control            | Bray–Curtis + PCoA + PERMANOVA |
| Alpha diversity × sex                 | Sex × Diagnosis                     | Factorial ANOVA                |
| Beta diversity × sex                  | Sex × Diagnosis                     | PERMANOVA + betadisper         |
| Sex-stratified alpha diversity        | Mood Disorder vs Control within sex | Wilcoxon rank-sum test         |
| Sex-stratified beta diversity         | Mood Disorder vs Control within sex | Bray–Curtis + PCoA + PERMANOVA |
| Differential abundance                | Mood Disorder vs Control            | Adjusted DESeq2                |
| Sex-stratified differential abundance | Mood Disorder vs Control within sex | Adjusted DESeq2                |
| Sex × diagnosis interaction           | Sex × Mood Disorder                 | DESeq2 interaction model       |

## Reproducibility

The scripts were developed using local file paths and therefore require modification of the input and output paths when run on another computer.

The analyses assume that the required phyloseq objects, metadata, and ASV taxonomy table are available in the corresponding local directories.

The `original_group_analyses` folder contains the earlier analyses based on the original BD, MDD, and Control classification.
