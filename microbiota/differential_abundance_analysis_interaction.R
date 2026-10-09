# Load libraries
##### Differential Abundance Analysis - Mood Disorder vs Control #####

# Load packages
library(dplyr)
library(phyloseq)
library(haven)
library(DESeq2)

# Load phyloseq object
ps <- readRDS("C:\\Users\\natal\\Documents\\data\\scripts\\objects\\phyloseq_object_adjusted.rds")

# Create grouping variable
sample_data(ps)$Mood_Disorder <- factor(
  ifelse(sample_data(ps)$thdico == 1, "Mood_Disorder", "Control"),
  levels = c("Control", "Mood_Disorder")
)

# Check group sizes
table(sample_data(ps)$Mood_Disorder)

# Sexo: categorical
sample_data(ps)$sexo <- factor(
  sample_data(ps)$sexo,
  levels = c(1, 2),
  labels = c("Male", "Female")
)

# Prepare covariates
# Continuous variables: centered and scaled
sample_data(ps)$IMC_c <- as.numeric(scale(sample_data(ps)$IMC))

# Categorical
# Psychiatric medication
sample_data(ps)$a20medicpsi <- factor(
  haven::zap_labels(sample_data(ps)$a20medicpsi),
  levels = c(0, 1),
  labels = c("Nao", "Sim")
)

# Antibiotics
sample_data(ps)$antib <- factor(
  haven::zap_labels(sample_data(ps)$antib),
  levels = c(0, 1),
  labels = c("Nao", "Sim")
)

# Prebiotics/probiotics/synbiotics
sample_data(ps)$suplem <- factor(
  haven::zap_labels(sample_data(ps)$suplem),
  levels = c(0, 1),
  labels = c("Nao", "Sim")
)

# ABEP
sample_data(ps)$ABEP <- factor(
  haven::zap_labels(sample_data(ps)$ABEP),
  levels = c(1, 2),
  labels = c("Alta", "Baixa")
)

# Check missings in variables
colSums(is.na(sample_data(ps)[, c(
  "ABEP",
  "a20medicpsi",
  "IMC",
  "antib",
  "suplem",
  "Mood_Disorder"
)]))

#### DESeq2 - SEX × MOOD DISORDER INTERACTION - ADJUSTED MODEL ####

# Exclude samples with missing values in the covariates
ids_excluir <- c(264, 492, 1049, 610)

ps_interaction_adj <- subset_samples(ps, !(id %in% ids_excluir))

# Remove samples with no remaining counts, if any
ps_interaction_adj <- prune_samples(sample_sums(ps_interaction_adj) > 0, ps_interaction_adj)

# Check number of samples
cat("Number of samples in adjusted interaction model:",nsamples(ps_interaction_adj),"\n")

# Add pseudocount
otu_table(ps_interaction_adj) <- otu_table(ps_interaction_adj) + 1

# DESeq2 interaction model adjusted for covariates
dds_interaction_adj <- phyloseq_to_deseq2(ps_interaction_adj, ~ ABEP + a20medicpsi + IMC_c + antib + suplem + sexo + Mood_Disorder + sexo:Mood_Disorder)
dds_interaction <- DESeq(dds_interaction_adj)
resultsNames(dds_interaction)

# Extract the SEX × MOOD DISORDER interaction
res_interaction <- results(dds_interaction,name = "sexoFemale.Mood_DisorderMood_Disorder", alpha = 0.05)
res_interaction <- res_interaction[order(res_interaction$padj, na.last = NA),]
res_interaction_df <- as.data.frame(res_interaction)

# Add ASV and taxonomy
res_interaction_df$ASV <- rownames(res_interaction_df)

taxonomy <- read.delim(
  "C:/Users/natal/Documents/data/scripts/tables/ASV_table.txt",
  header = TRUE,
  stringsAsFactors = FALSE,
  check.names = FALSE
)

res_interaction_df <- res_interaction_df %>%
  left_join(
    taxonomy %>%
      select(
        id,
        Kingdom,
        Phylum,
        Class,
        Order,
        Family,
        Genus,
        Species
      ),
    by = c("ASV" = "id")
  )


# Separate positive and negative interaction effects
res_interaction_up <- subset(res_interaction_df, padj < 0.05 & log2FoldChange > 0)
res_interaction_up <- res_interaction_up[order(res_interaction_up$log2FoldChange, decreasing = TRUE),]

res_interaction_down <- subset(res_interaction_df,padj < 0.05 & log2FoldChange < 0)
res_interaction_down <- res_interaction_down[order(res_interaction_down$log2FoldChange,decreasing = FALSE),]

#### Save results ####

options(scipen = 999)

results_path <- "C:\\Users\\natal\\Documents\\data\\scripts\\review 2\\results\\"


# Interaction results - positive effects
write.table(
  res_interaction_up,
  file = paste0(
    results_path,
    "deseq_THxC_sex_interaction_positive.txt"
  ),
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)


# Interaction results - negative effects
write.table(
  res_interaction_down,
  file = paste0(
    results_path,
    "deseq_THxC_sex_interaction_negative.txt"
  ),
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)


#### Top 5 positive and negative interaction effects ####
# Select top 5 based on original (unrounded) log2FoldChange
top5_interaction_up <- head(res_interaction_up,5)
top5_interaction_down <- head(res_interaction_down,5)

# Select columns
top5_interaction_up <- top5_interaction_up[, c(
  "ASV",
  "Kingdom",
  "Phylum",
  "Class",
  "Order",
  "Family",
  "Genus",
  "Species",
  "log2FoldChange",
  "padj"
)]

top5_interaction_down <- top5_interaction_down[, c(
  "ASV",
  "Kingdom",
  "Phylum",
  "Class",
  "Order",
  "Family",
  "Genus",
  "Species",
  "log2FoldChange",
  "padj"
)]

# Round log2FoldChange
top5_interaction_up$log2FoldChange <- round(top5_interaction_up$log2FoldChange,2)
top5_interaction_down$log2FoldChange <- round(top5_interaction_down$log2FoldChange,2)


# Format adjusted p-values
top5_interaction_up$padj <- ifelse(
  top5_interaction_up$padj < 0.001,
  "<0.001",
  formatC(
    top5_interaction_up$padj,
    format = "f",
    digits = 3
  )
)

top5_interaction_down$padj <- ifelse(
  top5_interaction_down$padj < 0.001,
  "<0.001",
  formatC(
    top5_interaction_down$padj,
    format = "f",
    digits = 3
  )
)

# Export Top 5 results
write.table(
  top5_interaction_up,
  file = paste0(
    results_path,
    "deseq_THxC_sex_interaction_top5_positive.txt"
  ),
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)

write.table(
  top5_interaction_down,
  file = paste0(
    results_path,
    "deseq_THxC_sex_interaction_top5_negative.txt"
  ),
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)
