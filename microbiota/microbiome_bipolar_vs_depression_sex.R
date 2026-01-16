# Load libraries
library(dada2)
library(phyloseq)
library(vegan)
library(ggplot2)
library(Biostrings)
library(tidyverse)

theme_set(theme_bw())

getwd()

options(scipen = 999)

# load objects
ps <- readRDS("C:\\Users\\natal\\Documents\\data\\scripts\\objects\\phyloseq_object_BXD.rds")
df_samples <- readRDS("C:\\Users\\natal\\Documents\\data\\scripts\\objects\\df_samples_BXD.rds")

#### Sex based analysis
# Subset the Phyloseq object by sex
df_samples$sexo
ps_female <- subset_samples(ps, Sex == 2)
ps_male <- subset_samples(ps, Sex == 1)

# Ensure we only keep bipolar and Depression groups in each subset
ps_female <- subset_samples(ps_female, Disorder %in% c("Depression", "Bipolar"))
ps_male <- subset_samples(ps_male, Disorder %in% c("Depression", "Bipolar"))

##### Alpha Diversity #####
# Estimar alpha diversidade
alpha_diversity_female <- estimate_richness(ps_female, measures = c("Observed", "Shannon", "InvSimpson"))
alpha_diversity_female <- cbind(alpha_diversity_female, sample_data(ps_female))
alpha_diversity_female$Sex <- "Female"

alpha_diversity_male <- estimate_richness(ps_male, measures = c("Observed", "Shannon", "InvSimpson"))
alpha_diversity_male <- cbind(alpha_diversity_male, sample_data(ps_male))
alpha_diversity_male$Sex <- "Male"

# Salvar os dados
write.table(alpha_diversity_female, file = "C:/Users/natal/Documents/data/scripts/results/alpha_diversity_female_BXD.txt", sep = "\t", row.names = TRUE, quote = FALSE)
write.table(alpha_diversity_male, file = "C:/Users/natal/Documents/data/scripts/results/alpha_diversity_male_BXD.txt", sep = "\t", row.names = TRUE, quote = FALSE)

# Testes de Wilcoxon
wilcox_results <- list(
  Female = list(
    Observed = wilcox.test(Observed ~ Disorder, data = alpha_diversity_female),
    Shannon = wilcox.test(Shannon ~ Disorder, data = alpha_diversity_female),
    InvSimpson = wilcox.test(InvSimpson ~ Disorder, data = alpha_diversity_female)
  ),
  Male = list(
    Observed = wilcox.test(Observed ~ Disorder, data = alpha_diversity_male),
    Shannon = wilcox.test(Shannon ~ Disorder, data = alpha_diversity_male),
    InvSimpson = wilcox.test(InvSimpson ~ Disorder, data = alpha_diversity_male)
  )
)

# Obter os p-valores
get_p <- function(sex, metric) {
  format(wilcox_results[[sex]][[metric]]$p.value, digits = 3, nsmall = 3)
}

# Unir os dados para plot
alpha_diversity_all <- rbind(alpha_diversity_female, alpha_diversity_male)
alpha_diversity_all$Sex <- factor(alpha_diversity_all$Sex, levels = c("Male", "Female"))

# Função para plotar
plot_alpha_diversity <- function(data, metric) {
  pvals <- data.frame(
    Sex = c("Male", "Female"),
    p_value = c(get_p("Male", metric), get_p("Female", metric))
  )
  
  ggplot(data, aes(x = Disorder, y = .data[[metric]], fill = Disorder)) +
    stat_boxplot(geom = "errorbar", position = position_dodge(0.5), width = 0.2) +
    geom_boxplot(alpha = 0.7, fatten = 2, lwd = 0.6, outlier.shape = NA) +
    geom_jitter(position = position_jitter(width = 0.2), alpha = 0.6, size = 1.5) +
    facet_wrap(~Sex) +
    geom_text(data = pvals,
              aes(x = 1.5, y = max(data[[metric]], na.rm = TRUE) * 1.05,  # posição do texto
                  label = paste0("p = ", p_value)),
              inherit.aes = FALSE,
              size = 4) +
    labs(title = paste(metric, "Alpha Diversity"),
         x = "Group", y = metric, fill = "Group") +
    theme_minimal() +
    theme(legend.position = "top") +
    scale_fill_manual(values = c("Depression" = "#79a2cd", "Bipolar" = "#e57a74"))
}

# Gerar os gráficos
plot_observed    <- plot_alpha_diversity(alpha_diversity_all, "Observed")
plot_shannon     <- plot_alpha_diversity(alpha_diversity_all, "Shannon")
plot_invsimpson  <- plot_alpha_diversity(alpha_diversity_all, "InvSimpson")

library(patchwork)
combined_plot <- plot_observed + plot_shannon + plot_invsimpson + plot_layout(ncol = 3)
print(combined_plot)

#### Beta diversity (Separado por Sexo) ####

## Distâncias
bray_dist_female <- phyloseq::distance(ps_female, method = "bray")
bray_dist_male   <- phyloseq::distance(ps_male, method = "bray")

#### PCoA ####
pcoa_bray_female     <- ordinate(ps_female, method = "PCoA", distance = bray_dist_female)
pcoa_bray_male       <- ordinate(ps_male,   method = "PCoA", distance = bray_dist_male)

## Função para PCoA
plot_pcoa <- function(ordination, ps_obj, title) {
  var_exp <- ordination$values$Relative_eig[1:2] * 100
  xlab <- paste0("PCoA 1 (", round(var_exp[1], 1), "%)")
  ylab <- paste0("PCoA 2 (", round(var_exp[2], 1), "%)")
  
  plot_ordination(ps_obj, ordination, color = "Disorder") +
    geom_point(size = 2, alpha = 0.8) +
    stat_ellipse(aes(group = Disorder, color = Disorder), level = 0.95, linewidth = 0.5) +
    labs(title = title, x = xlab, y = ylab) +
    theme_minimal() +
    scale_color_manual(values = c("Depression" = "#79a2cd", "Bipolar" = "#e57a74"))
}

# Gerar os gráficos por sexo
g_bray_female <- plot_pcoa(pcoa_bray_female, ps_female, "PCoA - Bray-Curtis (Female - Bipolar vs Depression)")
g_bray_male   <- plot_pcoa(pcoa_bray_male, ps_male, "PCoA - Bray-Curtis (Male - Bipolar vs Depression)")

print(g_bray_female)
print(g_bray_male)

#### NMDS ####
## NMDS por sexo - Bray-Curtis
nmds_bray_female <- metaMDS(bray_dist_female, k = 2, trymax = 100)
nmds_bray_male   <- metaMDS(bray_dist_male,   k = 2, trymax = 100)

## Função para plotar NMDS com elipses
plot_nmds <- function(nmds_obj, ps_obj, title) {
  scores_df <- as.data.frame(scores(nmds_obj, display = "sites"))
  scores_df$Disorder <- sample_data(ps_obj)$Disorder
  
  ggplot(scores_df, aes(x = NMDS1, y = NMDS2, color = Disorder)) +
    geom_point(size = 2, alpha = 0.8) +
    stat_ellipse(aes(group = Disorder, color = Disorder), level = 0.95, linewidth = 0.5) +
    labs(title = title, x = "NMDS 1", y = "NMDS 2") +
    theme_minimal() +
    scale_color_manual(values = c("Depression" = "#79a2cd", "Bipolar" = "#e57a74"))
}

## Gerar os gráficos por sexo e distância
g_nmds_bray_female <- plot_nmds(nmds_bray_female, ps_female, "NMDS - Bray-Curtis (Female - Bipolar vs Depression)")
g_nmds_bray_male   <- plot_nmds(nmds_bray_male,   ps_male,   "NMDS - Bray-Curtis (Male - Bipolar vs Depression)")

## Exibir se quiser
print(g_nmds_bray_female)
print(g_nmds_bray_male)

#### PERMANOVA e ANOSIM - Female ####
group_female <- factor(sample_data(ps_female)$Disorder)

# PERMANOVA
adonis_bray_f <- adonis2(bray_dist_female ~ group_female, permutations = 999)

# Extrair Model
df_bray_f <- as.data.frame(adonis_bray_f)

# Adicionar colunas
df_bray_f$Method <- "Bray-Curtis"
df_bray_f$Sex <-  "Female"

# ANOSIM
anosim_bray_f <- anosim(bray_dist_female, grouping = group_female, permutations = 999)


#### PERMANOVA e ANOSIM - Male ####
group_male <- factor(sample_data(ps_male)$Disorder)

# PERMANOVA
adonis_bray_m <- adonis2(bray_dist_male ~ group_male, permutations = 999)

# Extrair Model
df_bray_m <- as.data.frame(adonis_bray_m)

# Adicionar colunas
df_bray_m$Method <- "Bray-Curtis"
df_bray_m$Sex <- "Male"

# ANOSIM
anosim_bray_m <- anosim(bray_dist_male, grouping = group_male, permutations = 999)

#### Salvar resultados ####

# Padronizar colunas do PERMANOVA (caso necessário)
padronizar_colnames <- function(df) {
  colnames(df)[colnames(df) == "F.Model"] <- "F"
  colnames(df)[colnames(df) == "Pr(>F)"] <- "Pr(>F)"  # manter nome padrão
  df
}

# Padronizar DataFrame do PERMANOVA
df_bray_f <- padronizar_colnames(df_bray_f)
df_bray_m <- padronizar_colnames(df_bray_m)

# Adicionar colunas 'Method' e 'Sex'
df_bray_f$Method <- "Bray-Curtis"
df_bray_f$Sex <- "Female"

df_bray_m$Method <- "Bray-Curtis"
df_bray_m$Sex <- "Male"

# Combinar resultados do PERMANOVA
permanova_results <- rbind(df_bray_f, df_bray_m)

# Salvar PERMANOVA
write.table(permanova_results,
            file = "C:\\Users\\natal\\Documents\\data\\scripts\\results\\PERMANOVA_sexBXD.txt",
            sep = "\t", row.names = FALSE, quote = FALSE)

# Resultados do ANOSIM (já estão em data.frame limpo)
anosim_results_f <- data.frame(
  Method = "Bray-Curtis",
  R_statistic = anosim_bray_f$statistic,
  p_value = anosim_bray_f$signif,
  Sex = "Female"
)

anosim_results_m <- data.frame(
  Method = "Bray-Curtis",
  R_statistic = anosim_bray_m$statistic,
  p_value = anosim_bray_m$signif,
  Sex = "Male"
)

anosim_results_all <- rbind(anosim_results_f, anosim_results_m)

# Salvar ANOSIM
write.table(anosim_results_all,
            file = "C:\\Users\\natal\\Documents\\data\\scripts\\results\\ANOSIM_sexBXD.txt",
            sep = "\t", row.names = FALSE, quote = FALSE)


##### Differential Abundance DESeq #####
library(DESeq2)

# Female Differential Abundance
sample_data(ps_female)$Disorder <- as.factor(sample_data(ps_female)$Disorder)
sample_data(ps_female)$Disorder <- relevel(as.factor(sample_data(ps_female)$Disorder), ref = "Depression")
otu_table(ps_female) <- otu_table(ps_female) + 1 # objeto ps que salvamos não tinha +1
dds_female <- phyloseq_to_deseq2(ps_female, ~ Disorder)
dds_female <- DESeq(dds_female)

# Extract results
res_female <- results(dds_female, alpha = 0.05)

# Convert to data frame
res_female_df <- as.data.frame(res_female)

taxonomy <- as.data.frame(tax_table(ps_female))
res_female_df$ASV <- rownames(res_female_df)
res_female_df <- merge(res_female_df, taxonomy, by.x = "ASV", by.y = "row.names", all.x = TRUE)


# Separate upregulated and downregulated ASVs
res_female_up <- subset(res_female_df, padj < 0.05 & log2FoldChange > 0)
res_female_up <- res_female_up[order(res_female_up$log2FoldChange, decreasing = TRUE), ]
res_female_down <- subset(res_female_df, padj < 0.05 & log2FoldChange < 0)
res_female_down <- res_female_down[order(res_female_down$log2FoldChange, decreasing = FALSE), ]

# Save results
options(scipen = 999) # Avoid scientific notation

# Export as tab-separated .txt files
write.table(res_female_up, file = "C:\\Users\\natal\\Documents\\data\\scripts\\results\\deseq_female_BXD_up.txt", sep = "\t", row.names = FALSE, quote = FALSE)
write.table(res_female_down, file = "C:\\Users\\natal\\Documents\\data\\scripts\\results\\deseq_female_BXD_down.txt", sep = "\t", row.names = FALSE, quote = FALSE)

# Male Differential Abundance
sample_data(ps_male)$Disorder <- as.factor(sample_data(ps_male)$Disorder)
sample_data(ps_male)$Disorder <- relevel(as.factor(sample_data(ps_male)$Disorder), ref = "Depression")
otu_table(ps_male) <- otu_table(ps_male) + 1
dds_male <- phyloseq_to_deseq2(ps_male, ~ Disorder)
dds_male <- DESeq(dds_male)

# Extract results
res_male <- results(dds_male, alpha = 0.05)

# Convert to data frame
res_male_df <- as.data.frame(res_male)

taxonomy <- as.data.frame(tax_table(ps_male))
res_male_df$ASV <- rownames(res_male_df)
res_male_df <- merge(res_male_df, taxonomy, by.x = "ASV", by.y = "row.names", all.x = TRUE)

# Separate upregulated and downregulated ASVs
res_male_up <- subset(res_male_df, padj < 0.05 & log2FoldChange > 0)
res_male_up <- res_male_up[order(res_male_up$log2FoldChange, decreasing = TRUE), ]
res_male_down <- subset(res_male_df, padj < 0.05 & log2FoldChange < 0)
res_male_down <- res_male_down[order(res_male_down$log2FoldChange, decreasing = FALSE), ]

# Save results
options(scipen = 999) # Avoid scientific notation

# Export as tab-separated .txt files
write.table(res_male_up, file = "C:\\Users\\natal\\Documents\\data\\scripts\\results\\deseq_male_BXD_up.txt", sep = "\t", row.names = FALSE, quote = FALSE)
write.table(res_male_down, file = "C:\\Users\\natal\\Documents\\data\\scripts\\results\\deseq_male_BXD_down.txt", sep = "\t", row.names = FALSE, quote = FALSE)

##### Interpretação DESeq #####
levels(dds_female$Disorder)
levels(colData(dds_female)$Disorder)
levels(dds_male$Disorder)
levels(colData(dds_male)$Disorder)
# o primeiro nível listado é "Depression", isso significa que o grupo de 
# referência no DESeq2 é "Depression".
# Se o log2FoldChange é positivo, isso indica que a ASV é menos abundante no grupo
# de referência ("Depression") e mais abundante no grupo "Bipolar".
# Se o log2FoldChange é negativo, isso indica que a ASV é mais abundante no grupo
# de referência ("Depression") e menos abundante no grupo "Bipolar".


##### Gráficos ####
##### Volcano plot ######
##### Mulheres ####
# Remover NAs e calcular -log10(padj)
res_female_df <- res_female_df %>%
  filter(!is.na(padj)) %>%
  mutate(`-log10(padj)` = -log10(padj))

# Criar colunas de Significância
res_female_df$Significance <- case_when(
  res_female_df$padj < 0.05 & res_female_df$log2FoldChange > 0 ~ "Upregulated",
  res_female_df$padj < 0.05 & res_female_df$log2FoldChange < 0 ~ "Downregulated",
  TRUE ~ "Not Significant"
)

# Criar Volcano Plot
volcano_female_plot <- ggplot(res_female_df, aes(x = log2FoldChange, y = `-log10(padj)`, color = Significance)) +
  geom_point(alpha = 0.7, size = 3) +  # Pontos do gráfico
  scale_color_manual(values = c("Upregulated" = "#00bfc4", "Downregulated" = "#f8766d", "Not Significant" = "gray")) +
  geom_vline(xintercept = c(-1, 1), linetype = "dashed", color = "black") +  # Limites de significância
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "black") +  # Linha de corte para padj < 0.05
  labs(
    title = "Volcano Plot - Female - Bipolar vs Depression",
    x = "Log2 Fold Change",
    y = "-log10(padj)"
  ) +
  theme_minimal()

# Mostrar o gráfico
print(volcano_female_plot)


### Homens ####
#### Volcano Plot - Male ####

# Remover NAs e calcular -log10(padj)
res_male_df <- res_male_df %>%
  filter(!is.na(padj)) %>%
  mutate(`-log10(padj)` = -log10(padj))

# Criar colunas de Significância
res_male_df$Significance <- case_when(
  res_male_df$padj < 0.05 & res_male_df$log2FoldChange > 0 ~ "Upregulated",
  res_male_df$padj < 0.05 & res_male_df$log2FoldChange < 0 ~ "Downregulated",
  TRUE ~ "Not Significant"
)

# Criar Volcano Plot
volcano_male_plot <- ggplot(res_male_df, aes(x = log2FoldChange, y = `-log10(padj)`, color = Significance)) +
  geom_point(alpha = 0.7, size = 3) +  # Pontos do gráfico
  scale_color_manual(values = c("Upregulated" = "#00bfc4", "Downregulated" = "#f8766d", "Not Significant" = "gray")) +
  geom_vline(xintercept = c(-1, 1), linetype = "dashed", color = "black") +  # Limites de significância
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "black") +  # Linha de corte para padj < 0.05
  labs(
    title = "Volcano Plot - Male - Bipolar and Depression",
    x = "Log2 Fold Change",
    y = "-log10(padj)"
  ) +
  theme_minimal()

# Mostrar o gráfico
print(volcano_male_plot)