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

# Add pseudocount
otu_table(ps) <- otu_table(ps) + 1

#### DESeq2 - ADJUSTED MODEL ####
# Exclude samples with missing values in a20medicpsi or suplem
ids_excluir <- c(264, 492, 1049, 610)

ps_model2 <- subset_samples(ps, !(id %in% ids_excluir))

# Remove samples with no remaining counts, if any
ps_model2 <- prune_samples(sample_sums(ps_model2) > 0, ps_model2)

# Check number of samples
cat("Number of samples in Model 2:", nsamples(ps_model2), "\n")

# DESeq2 model
dds_adjusted_model2 <- phyloseq_to_deseq2(ps_model2, ~ ABEP + a20medicpsi + IMC_c + antib + suplem + Mood_Disorder)
dds_adjusted_model2 <- DESeq(dds_adjusted_model2)

res_adjusted_model2 <- results(dds_adjusted_model2, contrast = c("Mood_Disorder", 
                                                                 "Mood_Disorder",
                                                                  "Control"), alpha = 0.05
                                                                )

# Order by adjusted p-value
res_adjusted_model2 <- res_adjusted_model2[order(res_adjusted_model2$padj, na.last = NA),]

res_adjusted_model2_df <- as.data.frame(res_adjusted_model2)

# Add ASV and taxonomy
res_adjusted_model2_df$ASV <- rownames(res_adjusted_model2_df)

taxonomy <- read.delim(
  "C:/Users/natal/Documents/data/scripts/tables/ASV_table.txt",
  header = TRUE,
  stringsAsFactors = FALSE,
  check.names = FALSE
)

res_adjusted_model2_df <- res_adjusted_model2_df %>%
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

#### Separate upregulated and downregulated ASVs ####
res_adjusted_model2_up <- subset(res_adjusted_model2_df, padj < 0.05 & log2FoldChange > 0)
res_adjusted_model2_up <- res_adjusted_model2_up[order(res_adjusted_model2_up$log2FoldChange, decreasing = TRUE),]

res_adjusted_model2_down <- subset(res_adjusted_model2_df, padj < 0.05 & log2FoldChange < 0)
res_adjusted_model2_down <- res_adjusted_model2_down[order(res_adjusted_model2_down$log2FoldChange,decreasing = FALSE),]


#### Save results ####
options(scipen = 999)
results_path <- "C:\\Users\\natal\\Documents\\data\\scripts\\review 1\\results\\adjusted_DESeq2\\"


#### Save Adjusted Model results ####

write.table(res_adjusted_model2_up, file = paste0(results_path, "deseq_THxC_adjusted_up.txt"),
              sep = "\t",
              row.names = FALSE,
              quote = FALSE
            )

write.table(res_adjusted_model2_down, file = paste0(results_path, "deseq_THxC_adjusted_down.txt"),
            sep = "\t",
            row.names = FALSE,
            quote = FALSE
          )

#### Top 5 upregulated and downregulated ASVs ####
# Select top 5 based on original (unrounded) log2FoldChange
top5_up <- head(res_adjusted_model2_up, 5)
top5_down <- head(res_adjusted_model2_down, 5)


# Select and format columns 
top5_up <- top5_up[, c(
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

top5_down <- top5_down[, c(
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


#### Round values for export ####
top5_up$log2FoldChange <- round(top5_up$log2FoldChange,2)
top5_down$log2FoldChange <- round(top5_down$log2FoldChange,2)


# Format adjusted p-values
top5_up$padj <- ifelse(top5_up$padj < 0.001,"<0.001",formatC(top5_up$padj, format = "f", digits = 3))
top5_down$padj <- ifelse(top5_down$padj < 0.001,"<0.001",formatC(top5_down$padj, format = "f", digits = 3))


#### Export Top 5 ####
write.table(
  top5_up,
  file = paste0(results_path, "deseq_THxC_adjusted_top5_up.txt"),
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)

write.table(
  top5_down,
  file = paste0(results_path, "deseq_THxC_adjusted_top5_down.txt"),
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)
