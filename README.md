# Microbiome Study Analysis Repository

This repository contains scripts and notebooks for the analysis of a microbiome study, including sociodemographic and clinical data, microbiota analyses, and machine learning analyses.

The repository includes both the **original analyses based on the three diagnostic groups** and the **revised analyses based on the combined Mood Disorder vs Healthy Control grouping**.

## Folder Structure

### `/original_group_analyses`

Contains the original analysis files, which used the three-group diagnostic variable:

* Bipolar Disorder (BD)
* Major Depressive Disorder (MDD)
* Healthy Controls (HC)

These files were used for the initial analyses before the diagnostic groups were subsequently combined because of sample-size limitations.

The folder is organized into:

* `sociodemographic/` — R Markdown notebooks for sociodemographic and clinical analyses.
* `microbiota/` — R scripts for microbiota processing and statistical analyses.
* `machinelearning/` — Python scripts for machine learning analyses using the original diagnostic grouping.

### `/sociodemographic`

Contains the revised **R Markdown** notebooks for sociodemographic and clinical analyses using the combined grouping:

* Mood Disorders (BD or MDD)
* Healthy Controls (HC)

Includes data summaries, descriptive statistics, and plots.

### `/microbiota`

Contains the revised **R scripts** for microbiota analyses using the combined Mood Disorder vs Healthy Control grouping.

Analyses include:

* 16S rRNA sequencing data processing
* Alpha diversity analysis
* Beta diversity analysis
* Differential abundance analysis

The analyses use common microbiome analysis packages such as `phyloseq`, `vegan`, and `DESeq2`.

### `/machinelearning`

Contains the revised **Python scripts** for machine learning analyses using the combined **Mood Disorder vs Healthy Control** outcome.

The analyses include:

* Data preprocessing
* Feature preparation
* Model training
* Dimensionality reduction
* Class-imbalance handling
* Model evaluation and validation

## How to Use

1. Clone this repository:

   ```bash
   git clone <repository-url>
   ```

2. Navigate to the repository:

   ```bash
   cd <repository-name>
   ```

3. Refer to the README files and scripts within each folder for analysis-specific instructions.

## Analysis Grouping

The main analyses in this repository use the following grouping:

| Group           | Definition                                    |
| --------------- | --------------------------------------------- |
| Mood Disorder   | Bipolar Disorder or Major Depressive Disorder |
| Healthy Control | Participants without a mood disorder          |

The `original_group_analyses` folder contains analyses performed before this grouping was adopted and retains the original three-group classification (BD, MDD, and HC).

   git clone https://github.com/NWirowski/gut-microbiota-sex-analysis.git
