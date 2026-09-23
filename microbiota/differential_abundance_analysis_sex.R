##### Differential Abundance Analysis - Sex-stratified #####

# Load packages
library(dplyr)
library(phyloseq)
library(haven)
library(DESeq2)

# Load phyloseq object
ps <- readRDS("C:\\Users\\natal\\Documents\\data\\scripts\\objects\\phyloseq_object_adjusted.rds")

# Create grouping variable
sample_data(ps)$Mood_Disorder <- factor(
  ifelse(
    sample_data(ps)$thdico == 1,
    "Mood_Disorder",
    "Control"
  ),
  levels = c("Control", "Mood_Disorder")
)

# Check overall group sizes
table(sample_data(ps)$Mood_Disorder)

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

sample_data(ps)[
  is.na(sample_data(ps)$a20medicpsi) |
    is.na(sample_data(ps)$suplem),
  c("id", "sexo", "a20medicpsi", "suplem")
]

# Create sex-specific phyloseq objects
ps_female <- subset_samples(ps, sexo == 2)
ps_male <- subset_samples(ps, sexo == 1)

# Remove samples with zero reads, if any
ps_female <- prune_samples(sample_sums(ps_female) > 0, ps_female)

ps_male <- prune_samples(sample_sums(ps_male) > 0, ps_male)

# Check Mood Disorder vs Control distribution
table(sample_data(ps_female)$Mood_Disorder)
table(sample_data(ps_male)$Mood_Disorder)


# Add pseudocount
otu_table(ps_female) <- otu_table(ps_female) + 1
otu_table(ps_male) <- otu_table(ps_male) + 1

#### FEMALE ANALYSIS ####
#### FEMALE - ADJUSTED MODEL ####
# Exclude the two female samples with missing values
ids_excluir_fem <- c(1049, 610)

ps_female_M2 <- subset_samples(ps_female,!(id %in% ids_excluir_fem))

# Remove samples with zero reads, if any
ps_female_M2 <- prune_samples(sample_sums(ps_female_M2) > 0,ps_female_M2)

# DESeq2 - Female - Model 2
dds_female_adjusted_model2 <- phyloseq_to_deseq2(ps_female_M2, ~ ABEP + a20medicpsi + IMC_c + antib + suplem + Mood_Disorder)
dds_female_adjusted_model2 <- DESeq(dds_female_adjusted_model2)

res_female_adjusted_model2 <- results(dds_female_adjusted_model2,contrast = c("Mood_Disorder","Mood_Disorder","Control"), alpha = 0.05)
res_female_adjusted_model2 <- res_female_adjusted_model2[order(res_female_adjusted_model2$padj,na.last = NA),]
res_female_adjusted_model2_df <- as.data.frame(res_female_adjusted_model2)

# Add ASV and taxonomy
res_female_adjusted_model2_df$ASV <-rownames(res_female_adjusted_model2_df)

taxonomy <- read.delim(
  "C:/Users/natal/Documents/data/scripts/tables/ASV_table.txt",
  header = TRUE,
  stringsAsFactors = FALSE,
  check.names = FALSE
)

res_female_adjusted_model2_df <- res_female_adjusted_model2_df %>%
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

# Separate model 2 upregulated and downregulated ASVs
res_female_adjusted_model2_up <- subset(res_female_adjusted_model2_df,padj < 0.05 & log2FoldChange > 0)
res_female_adjusted_model2_up <- res_female_adjusted_model2_up[order(res_female_adjusted_model2_up$log2FoldChange,decreasing = TRUE),]

res_female_adjusted_model2_down <- subset(res_female_adjusted_model2_df,padj < 0.05 & log2FoldChange < 0)
res_female_adjusted_model2_down <- res_female_adjusted_model2_down[ order(res_female_adjusted_model2_down$log2FoldChange,decreasing = FALSE),]

#### MALE ANALYSIS ####
#### MALE - ADJUSTED MODEL ####
# Exclude the two male samples with missing values
ids_excluir_male <- c(264, 492)

ps_male_M2 <- subset_samples(ps_male,!(id %in% ids_excluir_male))

# Remove samples with zero reads, if any
ps_male_M2 <- prune_samples(sample_sums(ps_male_M2) > 0,ps_male_M2)

# DESeq2 - Male - Model 2
dds_male_adjusted_model2 <- phyloseq_to_deseq2(ps_male_M2, ~ ABEP + a20medicpsi + IMC_c + antib + suplem + Mood_Disorder)
dds_male_adjusted_model2 <- DESeq(dds_male_adjusted_model2)

res_male_adjusted_model2 <- results(dds_male_adjusted_model2,contrast = c("Mood_Disorder","Mood_Disorder","Control"),alpha = 0.05)
res_male_adjusted_model2 <- res_male_adjusted_model2[order(res_male_adjusted_model2$padj,na.last = NA),]
res_male_adjusted_model2_df <- as.data.frame(res_male_adjusted_model2)

# Add ASV and taxonomy
res_male_adjusted_model2_df$ASV <- rownames(res_male_adjusted_model2_df)

res_male_adjusted_model2_df <- res_male_adjusted_model2_df %>%
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


# Separate male model 2 upregulated and downregulated ASVs
res_male_adjusted_model2_up <- subset(res_male_adjusted_model2_df,padj < 0.05 & log2FoldChange > 0)
res_male_adjusted_model2_up <- res_male_adjusted_model2_up[order(res_male_adjusted_model2_up$log2FoldChange,decreasing = TRUE),]

res_male_adjusted_model2_down <- subset(res_male_adjusted_model2_df,padj < 0.05 & log2FoldChange < 0)
res_male_adjusted_model2_down <- res_male_adjusted_model2_down[order(res_male_adjusted_model2_down$log2FoldChange,decreasing = FALSE),]

#### Save results ####

options(scipen = 999)

results_path <- "C:\\Users\\natal\\Documents\\data\\scripts\\review 1\\results\\adjusted_DEseq2\\"

# Save Female Model 2 results

write.table(
  res_female_adjusted_model2_up,
  file = paste0(
    results_path,
    "deseq_femTHXC_M2_up.txt"
  ),
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)

write.table(
  res_female_adjusted_model2_down,
  file = paste0(
    results_path,
    "deseq_femTHXC_M2_down.txt"
  ),
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)

# Save Male Model 2 results

write.table(
  res_male_adjusted_model2_up,
  file = paste0(
    results_path,
    "deseq_maleTHXC_M2_up.txt"
  ),
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)

write.table(
  res_male_adjusted_model2_down,
  file = paste0(
    results_path,
    "deseq_maleTHXC_M2_down.txt"
  ),
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)

# Top 5 Female Model 2

top5_female_up <- head(
  res_female_adjusted_model2_up,
  5
)

top5_female_down <- head(
  res_female_adjusted_model2_down,
  5
)

# Select columns

top5_female_up <- top5_female_up[, c(
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

top5_female_down <- top5_female_down[, c(
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

top5_female_up$log2FoldChange <- round(
  top5_female_up$log2FoldChange,
  2
)

top5_female_down$log2FoldChange <- round(
  top5_female_down$log2FoldChange,
  2
)

#Format adjusted p-values

top5_female_up$padj <- ifelse(
  top5_female_up$padj < 0.001,
  "<0.001",
  formatC(
    top5_female_up$padj,
    format = "f",
    digits = 3
  )
)

top5_female_down$padj <- ifelse(
  top5_female_down$padj < 0.001,
  "<0.001",
  formatC(
    top5_female_down$padj,
    format = "f",
    digits = 3
  )
)

# Save Female Top 5

write.table(
  top5_female_up,
  file = paste0(
    results_path,
    "deseq_top5femTHXC_M2_up.txt"
  ),
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)

write.table(
  top5_female_down,
  file = paste0(
    results_path,
    "deseq_top5femTHXC_M2_down.txt"
  ),
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)

# Top 5 Male Model 2

top5_male_up <- head(
  res_male_adjusted_model2_up,
  5
)

top5_male_down <- head(
  res_male_adjusted_model2_down,
  5
)

# Select columns

top5_male_up <- top5_male_up[, c(
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

top5_male_down <- top5_male_down[, c(
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

top5_male_up$log2FoldChange <- round(
  top5_male_up$log2FoldChange,
  2
)

top5_male_down$log2FoldChange <- round(
  top5_male_down$log2FoldChange,
  2
)

# Format adjusted p-values

top5_male_up$padj <- ifelse(
  top5_male_up$padj < 0.001,
  "<0.001",
  formatC(
    top5_male_up$padj,
    format = "f",
    digits = 3
  )
)

top5_male_down$padj <- ifelse(
  top5_male_down$padj < 0.001,
  "<0.001",
  formatC(
    top5_male_down$padj,
    format = "f",
    digits = 3
  )
)

# Save Male Top 5

write.table(
  top5_male_up,
  file = paste0(
    results_path,
    "deseq_top5maleTHXC_M2_up.txt"
  ),
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)

write.table(
  top5_male_down,
  file = paste0(
    results_path,
    "deseq_top5maleTHXC_M2_down.txt"
  ),
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)