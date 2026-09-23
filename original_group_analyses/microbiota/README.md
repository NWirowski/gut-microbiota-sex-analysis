# Microbiota Analysis Scripts

This folder contains R scripts used for 16S rRNA microbiota analysis in the study comparing bipolar disorder, depression, and control groups. The scripts include **data preprocessing, diversity analyses, differential abundance, and visualization**.

## Folder Contents

### Scripts for 16S rRNA Data Processing

These scripts perform the full microbiota analysis starting from raw sequencing data:

| Script | Description |
|--------|-------------|
| `microbiome_bipolar_vs_control.R` | Full 16S processing and analysis comparing **Bipolar vs Control**. Includes filtering, denoising (DADA2), chimera removal, phyloseq object creation, alpha/beta diversity, PERMANOVA/ANOSIM, and differential abundance (DESeq2). |
| `microbiome_bipolar_vs_depression.R` | Same as above but comparing **Bipolar vs Depression**. |
| `microbiome_depression_vs_control.R` | Same as above but comparing **Depression vs Control**. |

### Scripts Using Preprocessed Phyloseq Objects

These scripts start from an **already created phyloseq object** (no raw 16S processing) and perform diversity and differential abundance analyses:

| Script | Description |
|--------|-------------|
| `microbiome_bipolar_vs_control_sex.R` | Diversity and differential abundance analyses for **Bipolar vs Control**, split by sex. Uses existing phyloseq object. |
| `microbiome_bipolar_vs_depression_sex.R` | Same as above but for **Bipolar vs Depression**, split by sex. |
| `microbiome_depression_vs_control_sex.R` | Same as above but for **Depression vs Control**, split by sex. |

---

## **Analysis Steps in Full 16S Scripts**

1. **Load metadata and raw FASTQ files**.
2. **Filter and trim sequences** using `filterAndTrim` (DADA2).
3. **Learn error rates** and **denoise sequences**.
4. **Remove chimeras** to generate a clean ASV table.
5. **Create phyloseq object** combining OTU/ASV table, taxonomy, and metadata.
6. **Alpha diversity analysis**: Observed, Shannon, and Inverse Simpson indices, with statistical tests (Shapiro, Wilcoxon).
7. **Beta diversity analysis**:
   - Bray-Curtis distance
   - PCoA and NMDS ordination
   - PERMANOVA and ANOSIM tests
8. **Differential abundance** using DESeq2
9. **Visualization**: Boxplots, volcano plots, PCoA/NMDS plots
10. **Save outputs and objects** for reuse in downstream analyses.

---

## **Dependencies**

The scripts use the following R packages:

```r
# Core microbiome analysis
library(dada2)
library(phyloseq)
library(DESeq2)
library(vegan)
library(Biostrings)
library(ggplot2)
library(tidyverse)
library(patchwork)
