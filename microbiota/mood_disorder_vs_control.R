# Load libraries
library(dada2)
library(phyloseq)
library(vegan)
library(ggplot2)
library(Biostrings)
library(tidyverse)
library(patchwork)

theme_set(theme_bw())

getwd()

options(scipen = 999)

# load objects
ps <- readRDS("C:\\Users\\natal\\Documents\\data\\scripts\\objects\\phyloseq_object_complete.rds")

#### Mood Disorder vs Control ####

#### Alpha Diversity#####

# Add grouping variable (mood disorder or control) as "factor" 
sample_data(ps)$Group <- factor(
  sample_data(ps)$thdico,
  levels = c(0, 1),
  labels = c("Control", "Mood Disorder")
)

# Estimate alpha diversity
alpha_diversity <- estimate_richness(ps, measures = c("Observed", "Shannon", "InvSimpson"))

# Add metadata
alpha_diversity <- cbind(alpha_diversity, sample_data(ps))

# Save results
write.table(
  alpha_diversity,
  file = "C:/Users/natal/Documents/data/scripts/review 1/results/alpha_diversity_THxC.txt",
  sep = "\t",
  row.names = TRUE,
  col.names = NA,
  quote = FALSE
)

# Mann-Whitney - test to see if there's statistical significance
wilcox_observed <- wilcox.test(Observed ~ Group, data = alpha_diversity)

wilcox_shannon <- wilcox.test(Shannon ~ Group, data = alpha_diversity)

wilcox_invsimpson <- wilcox.test(InvSimpson ~ Group, data = alpha_diversity)

# Obtaining the pvalues
p_observed <- format(wilcox_observed$p.value, digits = 3, nsmall = 3)

p_shannon <- format(wilcox_shannon$p.value, digits = 3, nsmall = 3)

p_invsimpson <- format(wilcox_invsimpson$p.value, digits = 3, nsmall = 3)


#### Alpha diversity plots ####

plot_alpha_diversity <- function(data, metric, p_value) {
  
  ggplot(
    data,
    aes(
      x = Group,
      y = .data[[metric]],
      fill = Group
    )
  ) +
    
    stat_boxplot(
      geom = "errorbar",
      position = position_dodge(0.5),
      width = 0.2
    ) +
    
    geom_boxplot(
      alpha = 0.7,
      fatten = 2,
      lwd = 0.6,
      outlier.shape = NA
    ) +
    
    geom_jitter(
      position = position_jitter(width = 0.2),
      alpha = 0.6,
      size = 1.5
    ) +
    
    annotate(
      "text",
      x = 1.5,
      y = max(data[[metric]], na.rm = TRUE) * 1.05,
      label = paste0("p = ", p_value),
      size = 4
    ) +
    
    labs(
      title = paste(metric, "Alpha Diversity"),
      x = "Group",
      y = metric,
      fill = "Group"
    ) +
    
    theme_minimal() +
    
    theme(
      legend.position = "top"
    ) +
    
    scale_fill_manual(
      values = c(
        "Control" = "#79a2cd",
        "Mood Disorder" = "#e57a74"
      )
    )
}

# Generating the plots

plot_observed <- plot_alpha_diversity(alpha_diversity, "Observed", p_observed)

plot_shannon <- plot_alpha_diversity(alpha_diversity, "Shannon", p_shannon)

plot_invsimpson <- plot_alpha_diversity(alpha_diversity, "InvSimpson", p_invsimpson)

# Combine plots

combined_plot <- plot_observed + plot_shannon + plot_invsimpson + plot_layout(ncol = 3)

print(combined_plot)

#### Alpha diversity - Sex × Diagnosis interaction ####

# Convert variables to factors
alpha_diversity$Sex <- factor(
  alpha_diversity$sexo,
  levels = c(1, 2),
  labels = c("Male", "Female")
)

alpha_diversity$Diagnosis <- factor(
  alpha_diversity$thdico,
  levels = c(0, 1),
  labels = c("Control", "Mood Disorder")
)

# Factorial ANOVA
anova_observed <- aov(
  Observed ~ Sex * Diagnosis,
  data = alpha_diversity
)

anova_shannon <- aov(
  Shannon ~ Sex * Diagnosis,
  data = alpha_diversity
)

anova_invsimpson <- aov(
  InvSimpson ~ Sex * Diagnosis,
  data = alpha_diversity
)

summary(anova_observed)
summary(anova_shannon)
summary(anova_invsimpson)

#### Beta diversity ####
# Calculating distaces
bray_dist <- phyloseq::distance(ps, method = "bray")

#### PCoA ####
# Rodar PCoA (principal coordinates analysis)
pcoa_bray <- ordinate(ps, method = "PCoA", distance = bray_dist)

# Creating function to plot PCoA with groups and %
plot_pcoa <- function(ordination, title) {
  # Extrair % variância dos eixos 1 e 2
  var_exp <- ordination$values$Relative_eig[1:2] * 100
  xlab <- paste0("PCoA 1 (", round(var_exp[1], 1), "%)")
  ylab <- paste0("PCoA 2 (", round(var_exp[2], 1), "%)")
  
  plot_ordination(ps, ordination, color = "Group") +
    geom_point(size = 2, alpha = 0.8) +                    # smaller dots
    stat_ellipse(aes(group = Group, color = Group),  # group elipse
                 level = 0.95, linewidth = 0.5) +          # finer line
    labs(title = title, x = xlab, y = ylab) +               # percentage %
    theme_minimal() +
    scale_color_manual(values = c("Control" = "#79a2cd", "Mood Disorder" = "#e57a74"))
}

# Create plots
p_pcoa_bray <- plot_pcoa(pcoa_bray, "PCoA - Bray-Curtis")
print(p_pcoa_bray)

#### PERMANOVA ####
# Test to assess whether beta diversity differs between groups

adonis_bray <- adonis2(
  bray_dist ~ Group,
  data = data.frame(sample_data(ps)),
  permutations = 999
)

print(adonis_bray)

# Extract results table
df_bray <- as.data.frame(adonis_bray)

# Export results as tab-separated .txt
write.table(
  df_bray,
  file = "C:/Users/natal/Documents/data/scripts/review 1/results/PERMANOVA_THxC.txt",
  sep = "\t",
  quote = FALSE,
  col.names = NA,
  row.names = TRUE
)


#### Alpha diversity stratified by sex ####

# Subset the Phyloseq object by sex
ps_female <- subset_samples(ps, sexo == 2)
ps_male <- subset_samples(ps, sexo == 1)

# Add grouping variable: Control vs Mood Disorder
sample_data(ps_female)$Group <- factor(
  sample_data(ps_female)$thdico,
  levels = c(0, 1),
  labels = c("Control", "Mood Disorder")
)

sample_data(ps_male)$Group <- factor(
  sample_data(ps_male)$thdico,
  levels = c(0, 1),
  labels = c("Control", "Mood Disorder")
)

##### Alpha Diversity #####

# Estimate alpha diversity - Female
alpha_diversity_female <- estimate_richness(ps_female, measures = c("Observed", "Shannon", "InvSimpson"))

alpha_diversity_female <- cbind(alpha_diversity_female, sample_data(ps_female))

alpha_diversity_female$Sex <- "Female"

# Estimate alpha diversity - Male
alpha_diversity_male <- estimate_richness(ps_male, measures = c("Observed", "Shannon", "InvSimpson"))

alpha_diversity_male <- cbind(alpha_diversity_male, sample_data(ps_male))

alpha_diversity_male$Sex <- "Male"

# Ensure grouping variable is a factor
alpha_diversity_female$Group <- factor(
  alpha_diversity_female$thdico,
  levels = c(0, 1),
  labels = c("Control", "Mood Disorder")
)

alpha_diversity_male$Group <- factor(
  alpha_diversity_male$thdico,
  levels = c(0, 1),
  labels = c("Control", "Mood Disorder")
)

# Save results
write.table(
  alpha_diversity_female,
  file = "C:/Users/natal/Documents/data/scripts/review 1/results/alpha_diversity_femTHxC.txt",
  sep = "\t",
  row.names = TRUE,
  col.names = NA,
  quote = FALSE
)

write.table(
  alpha_diversity_male,
  file = "C:/Users/natal/Documents/data/scripts/review 1/results/alpha_diversity_maleTHxC.txt",
  sep = "\t",
  row.names = TRUE,
  col.names = NA,
  quote = FALSE
)

# Mann-Whitney - test to see if there's statistical significance

wilcox_results <- list(
  Female = list(
    Observed = wilcox.test(
      Observed ~ Group,
      data = alpha_diversity_female
    ),
    Shannon = wilcox.test(
      Shannon ~ Group,
      data = alpha_diversity_female
    ),
    InvSimpson = wilcox.test(
      InvSimpson ~ Group,
      data = alpha_diversity_female
    )
  ),
  
  Male = list(
    Observed = wilcox.test(
      Observed ~ Group,
      data = alpha_diversity_male
    ),
    Shannon = wilcox.test(
      Shannon ~ Group,
      data = alpha_diversity_male
    ),
    InvSimpson = wilcox.test(
      InvSimpson ~ Group,
      data = alpha_diversity_male
    )
  )
)

# Obtain p-values
get_p <- function(sex, metric) {
  format(
    wilcox_results[[sex]][[metric]]$p.value,
    digits = 3,
    nsmall = 3
  )
}

# Save p-values
alpha_pvalues_sex <- data.frame(
  Sex = c(
    "Female", "Female", "Female",
    "Male", "Male", "Male"
  ),
  Metric = c(
    "Observed", "Shannon", "InvSimpson",
    "Observed", "Shannon", "InvSimpson"
  ),
  p_value = c(
    get_p("Female", "Observed"),
    get_p("Female", "Shannon"),
    get_p("Female", "InvSimpson"),
    get_p("Male", "Observed"),
    get_p("Male", "Shannon"),
    get_p("Male", "InvSimpson")
  )
)

write.table(
  alpha_pvalues_sex,
  file = "C:/Users/natal/Documents/data/scripts/review 1/results/alpha_diversity_pvalues_sex_THxC.txt",
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)

##### Alpha diversity plots #####

# Combine data for plotting
alpha_diversity_all <- rbind(
  alpha_diversity_female,
  alpha_diversity_male
)

alpha_diversity_all$Sex <- factor(
  alpha_diversity_all$Sex,
  levels = c("Male", "Female")
)

# Function to plot alpha diversity
plot_alpha_diversity <- function(data, metric) {
  
  pvals <- data.frame(
    Sex = c("Male", "Female"),
    p_value = c(
      get_p("Male", metric),
      get_p("Female", metric)
    )
  )
  
  ggplot(
    data,
    aes(
      x = Group,
      y = .data[[metric]],
      fill = Group
    )
  ) +
    
    stat_boxplot(
      geom = "errorbar",
      position = position_dodge(0.5),
      width = 0.2
    ) +
    
    geom_boxplot(
      alpha = 0.7,
      fatten = 2,
      lwd = 0.6,
      outlier.shape = NA
    ) +
    
    geom_jitter(
      position = position_jitter(width = 0.2),
      alpha = 0.6,
      size = 1.5
    ) +
    
    facet_wrap(~Sex) +
    
    geom_text(
      data = pvals,
      aes(
        x = 1.5,
        y = max(data[[metric]], na.rm = TRUE) * 1.05,
        label = paste0("p = ", p_value)
      ),
      inherit.aes = FALSE,
      size = 4
    ) +
    
    labs(
      title = paste(metric, "Alpha Diversity"),
      x = "Group",
      y = metric,
      fill = "Group"
    ) +
    
    theme_minimal() +
    
    theme(
      legend.position = "top"
    ) +
    
    scale_fill_manual(
      values = c(
        "Control" = "#79a2cd",
        "Mood Disorder" = "#e57a74"
      )
    )
}

# Generate plots
plot_observed <- plot_alpha_diversity(alpha_diversity_all,"Observed")

plot_shannon <- plot_alpha_diversity(alpha_diversity_all,"Shannon")

plot_invsimpson <- plot_alpha_diversity(alpha_diversity_all,"InvSimpson")

# Combine plots
combined_plot <- (plot_observed + plot_shannon + plot_invsimpson) + plot_layout(ncol = 3)

print(combined_plot)

#### Beta diversity stratified by sex ####

#### Female beta diversity #####

# Bray-Curtis distance
bray_dist_female <- phyloseq::distance(
  ps_female,
  method = "bray"
)

# PCoA
pcoa_bray_female <- ordinate(
  ps_female,
  method = "PCoA",
  distance = bray_dist_female
)

# PERMANOVA
adonis_bray_female <- adonis2(
  bray_dist_female ~ Group,
  data = data.frame(sample_data(ps_female)),
  permutations = 999
)

print(adonis_bray_female)


#### Male beta diversity #####

# Bray-Curtis distance
bray_dist_male <- phyloseq::distance(
  ps_male,
  method = "bray"
)

# PCoA
pcoa_bray_male <- ordinate(
  ps_male,
  method = "PCoA",
  distance = bray_dist_male
)

# PERMANOVA
adonis_bray_male <- adonis2(
  bray_dist_male ~ Group,
  data = data.frame(sample_data(ps_male)),
  permutations = 999
)

print(adonis_bray_male)


# Save PERMANOVA results

df_bray_female <- as.data.frame(adonis_bray_female)

df_bray_male <- as.data.frame(adonis_bray_male)

write.table(
  df_bray_female,
  file = "C:/Users/natal/Documents/data/scripts/review 1/results/PERMANOVA_female_THxC.txt",
  sep = "\t",
  quote = FALSE,
  col.names = NA,
  row.names = TRUE
)

write.table(
  df_bray_male,
  file = "C:/Users/natal/Documents/data/scripts/review 1/results/PERMANOVA_male_THxC.txt",
  sep = "\t",
  quote = FALSE,
  col.names = NA,
  row.names = TRUE
)


##### PCoA plots #####

# Function to plot PCoA
plot_pcoa_sex <- function(
    ps_object,
    ordination,
    title
) {
  
  # Extract percentage of variance explained
  var_exp <- ordination$values$Relative_eig[1:2] * 100
  
  xlab <- paste0(
    "PCoA 1 (",
    round(var_exp[1], 1),
    "%)"
  )
  
  ylab <- paste0(
    "PCoA 2 (",
    round(var_exp[2], 1),
    "%)"
  )
  
  plot_ordination(
    ps_object,
    ordination,
    color = "Group"
  ) +
    
    geom_point(
      size = 2,
      alpha = 0.8
    ) +
    
    stat_ellipse(
      aes(
        group = Group,
        color = Group
      ),
      level = 0.95,
      linewidth = 0.5
    ) +
    
    labs(
      title = title,
      x = xlab,
      y = ylab
    ) +
    
    theme_minimal() +
    
    scale_color_manual(
      values = c(
        "Control" = "#79a2cd",
        "Mood Disorder" = "#e57a74"
      )
    )
}

# Generate PCoA plots
p_pcoa_bray_female <- plot_pcoa_sex(
  ps_female,
  pcoa_bray_female,
  "PCoA - Bray-Curtis - Female"
)

p_pcoa_bray_male <- plot_pcoa_sex(
  ps_male,
  pcoa_bray_male,
  "PCoA - Bray-Curtis - Male"
)

# Print plots
print(p_pcoa_bray_female)
print(p_pcoa_bray_male)

#### Beta diversity - Sex × Diagnosis ####

metadata <- data.frame(sample_data(ps))

metadata$Sex <- factor(
  metadata$sexo,
  levels = c(1, 2),
  labels = c("Male", "Female")
)

metadata$Diagnosis <- factor(
  metadata$thdico,
  levels = c(0, 1),
  labels = c("Control", "Mood Disorder")
)

# PERMANOVA - Sex × Diagnosis

adonis_bray_interaction <- adonis2(
  bray_dist ~ Sex * Diagnosis,
  data = metadata,
  permutations = 999,
  by = "terms"
)

print(adonis_bray_interaction)

# Homogeneity of multivariate dispersion

group_sex_diagnosis <- interaction(
  metadata$Sex,
  metadata$Diagnosis
)

betadisper_interaction <- betadisper(
  bray_dist,
  group_sex_diagnosis
)

anova(betadisper_interaction)

# Convert PERMANOVA results to data frame
df_bray_interaction <- as.data.frame(adonis_bray_interaction)

# Save results
write.table(
  df_bray_interaction,
  file = "C:/Users/natal/Documents/data/scripts/review 1/results/alfa e beta diversidade/beta_diversity_interaction.txt",
  sep = "\t",
  quote = FALSE,
  col.names = NA,
  row.names = TRUE
)
