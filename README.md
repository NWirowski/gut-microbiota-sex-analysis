# Microbiome Study Analysis Repository

This repository contains scripts and notebooks for the analysis of a microbiome study, including sociodemographic/clinical data, microbiota processing, and machine learning analyses. The project is organized into three main folders:

## Folder Structure

### `/sociodemographic`
- Contains **R Markdown** notebooks with basic analyses of sociodemographic and clinical features.
- Includes data summaries, descriptive statistics, and plots.

### `/microbiota`
- Contains **R scripts** for processing 16S rRNA sequencing data.
- Includes scripts for:
  - Alpha diversity analysis
  - Beta diversity analysis
  - Differential abundance analysis
- Designed to work with common microbiome analysis packages such as `phyloseq`, `vegan`, and `DESeq2`.

### `/machinelearning`
- Contains **Python scripts** for machine learning analyses on microbiome and clinical data.
- Includes preprocessing, model training, and evaluation scripts.

## How to Use
1. Clone this repository:
   ```bash
   git clone https://github.com/NWirowski/gut-microbiota-sex-analysis.git
