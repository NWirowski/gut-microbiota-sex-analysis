#!/usr/bin/Rscript

# Install required packages if not already installed
# required_packages <- c(
#  "BiocManager", "phyloseq", "dada2", "DESeq2", 
#  "vegan", "ggplot2", "Biostrings", "tidyverse"
# )

# install_if_missing <- function(packages) {
#  to_install <- packages[!packages %in% installed.packages()[, "Package"]]
#  if (length(to_install) > 0) {
#    install.packages(to_install, dependencies = TRUE)
#  }
# }

# install_if_missing(c("BiocManager", "tidyverse"))
# BiocManager::install(
#  c("phyloseq", "dada2", "DESeq2", "vegan", "ggplot2", "Biostrings"), 
#  ask = FALSE, update = FALSE
# )

# Load libraries
library(dada2)
library(phyloseq)
library(vegan)
library(ggplot2)
library(Biostrings)
library(tidyverse) 
library(patchwork)

theme_set(theme_bw())

#### Load sample metadata ####
# file.choose()
metadata_file <- "C:\\Users\\natal\\Documents\\data\\metadata.txt"
if (!file.exists(metadata_file)) {
  stop("Metadata file not found.")
}

df_samples <- as_tibble(read.csv2(file = "C:\\Users\\natal\\Documents\\data\\metadata.txt", sep = "\t"))
df_samples

# fastq data
# choose.dir()
raw_path <- "C:\\Users\\natal\\Documents\\data\\fastq"
if (!dir.exists(raw_path)) {
  stop("Raw FASTQ directory not found.")
}

# Listar arquivos fastq
fastq_files <- list.files(raw_path, pattern = ".fastq$", full.names = TRUE)

# Extrair os nomes base (sem extensão)
fastq_ids <- sapply(strsplit(basename(fastq_files), "\\."), `[`, 1)

# Criar um data.frame com nomes e caminhos
df_fastq <- data.frame(id = fastq_ids, fastq_path = fastq_files, stringsAsFactors = FALSE)
df_samples$id <- as.character(df_samples$id)
df_fastq$id <- as.character(df_fastq$id)
head(df_fastq)

# Juntar metadata com caminhos corretos, pelo id
df_samples <- df_samples %>%
  left_join(df_fastq, by = "id")

# Agora, seu df_samples tem a coluna fastq_path com o caminho correto para cada amostra


# Adjust dataframe to analyze disorder vs. control: bip
df_samples <- df_samples %>% 
    filter(mood %in% c(0,2)) %>% 
    mutate(group = if_else(mood != 0, "Bipolar", "Control"))

# Create directory for filtered data
# choose.dir()
filtered_dir <- "C:\\Users\\natal\\Documents\\data\\filtered"
if (!dir.exists(filtered_dir)) dir.create(filtered_dir, recursive = TRUE)

df_samples <- df_samples %>%
  mutate(fastqs_filtered = file.path(filtered_dir, paste0(id, "_filtered.fastq.gz")))

df_samples <- drop_na(df_samples)
df_samples$group

#### filtering ####
# Filter and trim sequences
filtered_sequences <- filterAndTrim(df_samples$fastq_path, df_samples$fastqs_filtered, 
                                    trimLeft = c(19), truncLen = c(240), maxN = 0, 
                                    maxEE = c(2), truncQ = 2, rm.phix = TRUE, 
                                    compress = TRUE, multithread = TRUE
                                    )

saveRDS(filtered_sequences, file = "C:\\Users\\natal\\Documents\\data\\scripts\\objects\\filtered_sequences_BXC.rds")

#### Learn error rates ####
errF <- learnErrors(df_samples$fastqs_filtered, multithread = TRUE)
plotErrors(errF, nominalQ = TRUE)

saveRDS(errF, file = "C:\\Users\\natal\\Documents\\data\\scripts\\objects\\errF_BXC.rds")

#### Denoising ####
dada_reads <- dada(df_samples$fastqs_filtered, err = errF, multithread = TRUE)
seqtab <- makeSequenceTable(dada_reads)

saveRDS(dada_reads, file = "C:\\Users\\natal\\Documents\\data\\scripts\\objects\\dada_reads_BXC.rds")

#### Remove chimeras ####
seqtab.nochim <- removeBimeraDenovo(seqtab, method = "consensus", multithread = TRUE, verbose = TRUE)

saveRDS(seqtab.nochim, file = "C:\\Users\\natal\\Documents\\data\\scripts\\objects\\seqtab.nochim_BXC.rds")


#### Conferências de filtragem ####

# Proporção total de sequências mantidas após remoção de quimeras
prop_mantidas <- sum(seqtab.nochim) / sum(seqtab)
cat("Proporção total de sequências mantidas após remoção de quimeras:", round(prop_mantidas, 3), "\n")

writeLines(
  paste("Proporção total de sequências mantidas após remoção de quimeras:", round(prop_mantidas, 3)),
  con = "C:\\Users\\natal\\Documents\\data\\scripts\\results\\proporcao_mantida_BXC.txt"
)

# Avaliar o número de sequências perdidas em cada etapa
getN <- function(x) sum(getUniques(x))

track <- cbind(
  input = filtered_sequences[, "reads.in"],
  filtered = filtered_sequences[, "reads.out"],
  denoised = sapply(dada_reads, getN),
  nonchim = rowSums(seqtab.nochim)
)

rownames(track) <- sample.names
track_df <- as.data.frame(track)

write.table(track_df,
            file = "C:\\Users\\natal\\Documents\\data\\scripts\\results\\track_BXC.txt",
            sep = "\t", quote = FALSE, row.names = TRUE, col.names = NA)

# Taxonomic assignment
taxa <- assignTaxonomy(seqtab.nochim, "C:\\Users\\natal\\Documents\\data\\silva_nr99_v138.2_toGenus_trainset.fa.gz", multithread=TRUE)
taxa <- addSpecies(taxa, "C:\\Users\\natal\\Documents\\data\\silva_v138.2_assignSpecies.fa.gz")
taxa

saveRDS(taxa, file = "C:\\Users\\natal\\Documents\\data\\scripts\\objects\\taxa_BXC.rds")

# Create Phyloseq object
samdf <- df_samples %>%
  select(id, group, sexo, corpele, idade, anosestudo, ABEP, histpsifam, thdico, ansiedadeatual, gravidade, gravidade_cat) %>%
  rename(Subject = id, Disorder = group, Sex = sexo) %>%
  column_to_rownames(var = "Subject")
rownames(samdf) <- paste(rownames(samdf), "filtered.fastq.gz", sep = "_")

ps <- phyloseq(
  otu_table(seqtab.nochim, taxa_are_rows = FALSE),
  sample_data(samdf),
  tax_table(taxa)
)

result <- apply(otu_table(ps), 1, function(x) all(x > 0))
str(result)
length(result)

taxa_to_keep <- apply(otu_table(ps), 2, function(x) all(x > 0))
ps_pruned <- prune_taxa(taxa_to_keep, ps)

# Assign DNA sequences to taxa
dna <- Biostrings::DNAStringSet(taxa_names(ps))
names(dna) <- taxa_names(ps)
ps <- merge_phyloseq(ps, dna)
taxa_names(ps) <- paste0("ASV", seq(ntaxa(ps)))

##### SAVE THE PHYLOSEQ OBJECT ####
saveRDS(ps, file = "C:\\Users\\natal\\Documents\\data\\scripts\\objects\\phyloseq_object_BXC.rds")

##### Alpha Diversity #####
alpha_diversity <- estimate_richness(ps, measures = c("Observed", "Shannon", "InvSimpson"))
alpha_diversity <- cbind(alpha_diversity, sample_data(ps))
write.table(alpha_diversity, file = "C:\\Users\\natal\\Documents\\data\\scripts\\results\\alpha_diversity_BXC.txt", sep = "\t", row.names = TRUE, quote = FALSE)

# Teste de normalidade #
# Rodar os testes
shapiro_observed <- shapiro.test(alpha_diversity$Observed)
shapiro_shannon <- shapiro.test(alpha_diversity$Shannon)
shapiro_invsimpson <- shapiro.test(alpha_diversity$InvSimpson)

# Criar um data frame com os resultados
shapiro_results <- data.frame(
  Metric = c("Observed", "Shannon", "InvSimpson"),
  W = c(
    round(shapiro_observed$statistic, 4),
    round(shapiro_shannon$statistic, 4),
    round(shapiro_invsimpson$statistic, 4)
  ),
  p_value = c(
    format(shapiro_observed$p.value, digits = 3, nsmall = 3),
    format(shapiro_shannon$p.value, digits = 3, nsmall = 3),
    format(shapiro_invsimpson$p.value, digits = 3, nsmall = 3)
  )
)

# Exportar para arquivo .txt separado por tabulação
write.table(shapiro_results,
            file = "C:\\Users\\natal\\Documents\\data\\scripts\\results\\shapiro_alpha_diversity_BXC.txt",
            sep = "\t", 
            row.names = FALSE,
            quote = FALSE)


# Wilcoxon test for each diversity metric
observed_test <- wilcox.test(Observed ~ Disorder, data = alpha_diversity)
shannon_test <- wilcox.test(Shannon ~ Disorder, data = alpha_diversity)
invsimpson_test <- wilcox.test(InvSimpson ~ Disorder, data = alpha_diversity)

# Extrair p-valores formatados com 3 casas decimais
observed_pvalue <- format(observed_test$p.value, digits = 3, nsmall = 3)
shannon_pvalue <- format(shannon_test$p.value, digits = 3, nsmall = 3)
invsimpson_pvalue <- format(invsimpson_test$p.value, digits = 3, nsmall = 3)

# Create a dataframe with p-values
# Criar data frame com os p-valores formatados
alpha_pvalues <- data.frame(
  Metric = c("Observed", "Shannon", "InvSimpson"),
  p_value = c(observed_pvalue, shannon_pvalue, invsimpson_pvalue)
)

# Exportar para arquivo tab-separated (.txt)
write.table(alpha_pvalues, 
            file = "C:\\Users\\natal\\Documents\\data\\scripts\\results\\alpha_diversity_pvalues_BXC.txt", 
            sep = "\t", 
            row.names = FALSE, 
            quote = FALSE)

# Boxplot of Shannon diversity by group
alpha_diversity$Disorder <- factor(alpha_diversity$Disorder, levels = c("Control", "Bipolar"))
pd <- position_dodge(width = 0.5)

plot_alpha_diversity <- function(data, metric, pval) {
  ymax <- max(data[[metric]]) * 1.05  # para posicionar o texto acima dos boxplots
  ggplot(data, aes(x = Disorder, y = .data[[metric]], fill = Disorder)) +
    stat_boxplot(geom = "errorbar", position = pd, width = 0.2) +
    geom_boxplot(alpha = 0.7, fatten = 2, size = 0.6) +
    geom_jitter(position = position_jitter(width = 0.2), alpha = 0.5, size = 1.5) +
    annotate("text", x = 1.5, y = ymax, label = paste0("p = ", pval), size = 5) +
    labs(title = paste(metric, "Alpha Diversity"), x = "Group", y = metric) +
    theme_minimal() +
    theme(legend.position = "none") +
    scale_fill_manual(values = c("Control" = "#79a2cd", "Bipolar" = "#e57a74"))
}

plot_observed <- plot_alpha_diversity(alpha_diversity, "Observed", observed_pvalue)
plot_shannon <- plot_alpha_diversity(alpha_diversity, "Shannon", shannon_pvalue)
plot_invsimpson <- plot_alpha_diversity(alpha_diversity, "InvSimpson", invsimpson_pvalue)

library(patchwork)
combined_plot <- plot_observed + plot_shannon + plot_invsimpson + plot_layout(ncol = 3)
print(combined_plot)


#### Beta diversity ####
# Calcular as distâncias
bray_dist <- phyloseq::distance(ps, method = "bray")

#### PCoA ####
# Rodar PCoA (principal coordinates analysis)
pcoa_bray <- ordinate(ps, method = "PCoA", distance = bray_dist)

# Criar função para plotar PCoA com grupos
plot_pcoa <- function(ordination, title) {
  # Extrair % variância dos eixos 1 e 2
  var_exp <- ordination$values$Relative_eig[1:2] * 100
  xlab <- paste0("PCoA 1 (", round(var_exp[1], 1), "%)")
  ylab <- paste0("PCoA 2 (", round(var_exp[2], 1), "%)")
  
  plot_ordination(ps, ordination, color = "Disorder") +
    geom_point(size = 2, alpha = 0.8) +                    # pontos menores
    stat_ellipse(aes(group = Disorder, color = Disorder),  # elipse por grupo
                 level = 0.95, linewidth = 0.5) +          # linha mais fina
    labs(title = title, x = xlab, y = ylab) +               # rótulos com %
    theme_minimal() +
    scale_color_manual(values = c("Control" = "#79a2cd", "Bipolar" = "#e57a74"))
}

# Gerar os gráficos
p_pcoa_bray <- plot_pcoa(pcoa_bray, "PCoA - Bray-Curtis")
print(p_pcoa_bray)

#### PERMANOVA ####
# Extrair o vetor do fator que quer testar (ex: Disorder)
group_factor <- factor(sample_data(ps)$Disorder)

# Rodar PERMANOVA para Bray-Curtis
adonis_bray <- adonis2(bray_dist ~ group_factor, permutations = 999)
print(adonis_bray)

# Extrair tabelas dos resultados
df_bray <- as.data.frame(adonis_bray)

# Exportar para arquivo .txt separado por tab
write.table(df_bray, file = "C:\\Users\\natal\\Documents\\data\\scripts\\results\\PERMANOVA_BXC.txt", sep = "\t", quote = FALSE, row.names = FALSE)

#### NMDS ####
# Rodar NMDS (non-metric multidimensional scaling)
nmds_bray <- metaMDS(bray_dist, k = 2, trymax = 100)

# Criar função para plotar NMDS com grupos
plot_nmds <- function(nmds_obj, title) {
  # Extrair scores para os eixos 1 e 2
  scores_df <- as.data.frame(scores(nmds_obj, display = "sites"))
  scores_df$Disorder <- sample_data(ps)$Disorder
  
  ggplot(scores_df, aes(x = NMDS1, y = NMDS2, color = Disorder)) +
    geom_point(size = 2, alpha = 0.8) +
    stat_ellipse(aes(group = Disorder, color = Disorder), level = 0.95, linewidth = 0.5) +
    labs(title = title, x = "NMDS 1", y = "NMDS 2") +
    theme_minimal() +
    scale_color_manual(values = c("Control" = "#79a2cd", "Bipolar" = "#e57a74"))
}

# Gerar os gráficos
p_nmds_bray <- plot_nmds(nmds_bray, "NMDS - Bray-Curtis")

print(p_nmds_bray)


#### ANOSIM ####
# Rodar ANOSIM para Bray-Curtis
anosim_bray <- anosim(bray_dist, grouping = group_factor, permutations = 999)
print(anosim_bray)


# Criar data.frame com os resultados ANOSIM
anosim_results <- data.frame(
  Dissimilarity = "Bray-Curtis",
  Statistic_R = round(anosim_bray$statistic, 4),
  p_value = format(anosim_bray$signif, digits = 3, nsmall = 3),
  Permutations = anosim_bray$permutations
)

# Exportar para arquivo .txt separado por tab
write.table(anosim_results,
            file = "C:\\Users\\natal\\Documents\\data\\scripts\\results\\ANOSIM_BXC.txt",
            sep = "\t", quote = FALSE, row.names = FALSE)


##### Differential Abundance Analysis #####
library(DESeq2)

# Set 'control' as the reference level for 'Bipolar'
sample_data(ps)$Disorder <- as.factor(sample_data(ps)$Disorder)
sample_data(ps)$Disorder <- relevel(sample_data(ps)$Disorder, ref = "Control")
otu_table(ps) <- otu_table(ps) + 1
dds <- phyloseq_to_deseq2(ps, ~ Disorder)
dds <- DESeq(dds)

res <- results(dds, alpha = 0.05)
res <- res[order(res$padj, na.last = NA), ]
res_df <- as.data.frame(res)

taxonomy <- as.data.frame(tax_table(ps))
res_df$ASV <- rownames(res_df)
res_df <- merge(res_df, taxonomy, by.x = "ASV", by.y = "row.names", all.x = TRUE)


# Separate upregulated and downregulated ASVs and order by log2FC
res_up <- subset(res_df, padj < 0.05 & log2FoldChange > 0)
res_up <- res_up[order(res_up$log2FoldChange, decreasing = TRUE), ]
res_down <- subset(res_df, padj < 0.05 & log2FoldChange < 0)
res_down <- res_down[order(res_down$log2FoldChange, decreasing = FALSE), ]

# Save results
options(scipen = 999) # Avoid scientific notation

# Export as tab-separated .txt files
write.table(res_up, file = "C:\\Users\\natal\\Documents\\data\\scripts\\results\\deseq_BXC_up.txt", sep = "\t", row.names = FALSE, quote = FALSE)
write.table(res_down, file = "C:\\Users\\natal\\Documents\\data\\scripts\\results\\deseq_BXC_down.txt", sep = "\t", row.names = FALSE, quote = FALSE)

#### Volcano Plot ####
# Preparar os dados para o volcano plot
res_df_clean <- res_df %>%
  filter(!is.na(padj)) %>%                         # Remove NAs em padj
  mutate(`-log10(padj)` = -log10(padj)) %>%       # Calcula -log10(padj)
  mutate(Significance = case_when(                 # Define categorias para cores
    padj < 0.05 & log2FoldChange > 0 ~ "Upregulated",
    padj < 0.05 & log2FoldChange < 0 ~ "Downregulated",
    TRUE ~ "Not Significant"
  ))

# Plot Volcano
volcano_plot <- ggplot(res_df_clean, aes(x = log2FoldChange, y = `-log10(padj)`, color = Significance)) +
  geom_point(alpha = 0.7, size = 3) +
  scale_color_manual(values = c("Upregulated" = "#00bfc4", "Downregulated" = "#f8766d", "Not Significant" = "gray")) +
  geom_vline(xintercept = c(-1, 1), linetype = "dashed", color = "black") +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "black") +
  labs(
    title = "Volcano Plot - Bipolar vs Control",
    x = "Log2 Fold Change",
    y = "-log10(padj)"
  ) +
  theme_minimal()

# Mostrar o gráfico
print(volcano_plot)


#### SAVE OBJECTS ####
# saveRDS(df_samples, file = "C:\\Users\\natal\\Documents\\data\\scripts\\objects\\df_samples_BXC.rds")
# saveRDS(filtered_sequences, file = "C:\\Users\\natal\\Documents\\data\\scripts\\objects\\filtered_sequences_BXC.rds")
# saveRDS(errF, file = "C:\\Users\\natal\\Documents\\data\\scripts\\objects\\errF_BXC.rds")
# saveRDS(dada_reads, file = "C:\\Users\\natal\\Documents\\data\\scripts\\objects\\dada_reads_BXC.rds")
# saveRDS(seqtab.nochim, file = "C:\\Users\\natal\\Documents\\data\\scripts\\objects\\seqtab.nochim_BXC.rds")
# saveRDS(taxa, file = "C:\\Users\\natal\\Documents\\data\\scripts\\objects\\taxa_BXC.rds")
# saveRDS(taxa.print, file = "C:\\Users\\natal\\Documents\\data\\scripts\\objects\\taxa.print_BXC.rds")
# saveRDS(samdf, file = "C:\\Users\\natal\\Documents\\data\\scripts\\objects\\samdf_BXC.rds")
# saveRDS(ps, file = "C:\\Users\\natal\\Documents\\data\\scripts\\objects\\phyloseq_object_BXC.rds")
# saveRDS(ps, file = "C:\\Users\\natal\\Documents\\data\\scripts\\objects\\phyloseq_object_tree_BXC.rds")
# saveRDS(dna, file = "C:\\Users\\natal\\Documents\\data\\scripts\\objects\\dna_BXC.rds")
# saveRDS(fastqs_filtered, file = "C:\\Users\\natal\\Documents\\data\\scripts\\objects\\fastqs_filtered_BXC.rds")
# saveRDS(sample.names, file = "C:\\Users\\natal\\Documents\\data\\scripts\\objects\\sample.names_BXC.rds")
# saveRDS(samples.out, file = "C:\\Users\\natal\\Documents\\data\\scripts\\objects\\samples.out_BXC.rds")

#### LOAD OBJECTS ####
# file.choose()
dada_reads <- readRDS("C:\\Users\\natal\\Documents\\data\\scripts\\objects\\dada_reads_BXC.rds")
df_samples <- readRDS("C:\\Users\\natal\\Documents\\data\\scripts\\objects\\df_samples_BXC.rds")
errF <- readRDS("C:\\Users\\natal\\Documents\\data\\scripts\\objects\\errF_BXC.rds")
filtered_sequences <- readRDS("C:\\Users\\natal\\Documents\\data\\scripts\\objects\\filtered_sequences_BXC.rds")
ps <- readRDS("C:\\Users\\natal\\Documents\\data\\scripts\\objects\\phyloseq_object_BXC.rds")
samdf <- readRDS("C:\\Users\\natal\\Documents\data\\scripts\\objects\\samdf_BXC.rds")
seqtab.nochim <- readRDS("C:\\Users\\natal\\Documents\\data\\scripts\\objects\\seqtab.nochim_BXC.rds")
taxa <- readRDS("C:\\Users\\natal\\Documents\\data\\scripts\\objects\\taxa_BXC.rds")