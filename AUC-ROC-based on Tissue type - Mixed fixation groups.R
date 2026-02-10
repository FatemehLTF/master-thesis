
# AUC-ROC: ONE-VS-REST (m/z per tissue) / based on different tissue types
# Combined across all fixation times!
# Repeated once for Trypsin (peptide dataset) and once for PNGase F (Glycan dataset)

# this code run for Glycan (PNGase F data set)


library(tidyverse)
library(pROC)
library(patchwork)
library(cowplot)
library(grid)

# Color base on tissue types
tissue_colors <- c(
  "brain"    = "#0000FF",
  "kidney"   = "#808080",
  "liver"    = "#800080",
  "lung"     = "#008000",
  "muscle"   = "#FF0000",
  "pancreas" = "#8B4513",
  "spleen"   = "#FFFF00"
)

# Load data
df_wide <- readRDS("S:/266_and_267/Roc_PLOT_final/TT/ROC_PNGase_effect of TT/all_data_df_PNGase_combine_267and266.rds")

# Convert dataset to long format
df_long <- df_wide %>%
  pivot_longer(
    cols = starts_with("m.z."),
    names_to = "mz",
    values_to = "intensity"
  ) %>%
  mutate(
    mz = sub("^m\\.z\\.", "", mz),
    tissue_type = as.character(tissue_type),
    fixation_time = as.character(fixation_time),
    Experiment = as.character(Experiment),
    intensity = as.numeric(intensity)
  ) %>%
  filter(
    !is.na(intensity),
    !is.na(Experiment),
    Experiment %in% c("266", "267"),
    tissue_type %in% names(tissue_colors)
  )


# one slide roc-plot (all tissues vs rest)

plot_roc_one_exp <- function(df_long, mzf, exp_id) {
  df_sub <- df_long %>%
    filter(
      mz == !!as.character(mzf),
      Experiment == !!exp_id
    )
  
  if (nrow(df_sub) == 0) {
    warning(paste("No data for exp", exp_id, "mz", mzf))
    return(ggplot() + theme_void() + ggtitle(paste("Exp", exp_id, "- No Data")))
  }
  
  # Base ROC background plot
  p <- ggplot() +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray70", size = 0.6) +
    geom_hline(yintercept = seq(0, 1, by = 0.2), color = "gray92", size = 0.3) +
    geom_vline(xintercept = seq(0, 1, by = 0.2), color = "gray92", size = 0.3) +
    theme_minimal(base_size = 11) +
    theme(
      plot.title = element_text(size = 12, face = "bold", hjust = 0.5),
      axis.title = element_text(size = 10),
      axis.text = element_text(size = 9),
      panel.grid = element_blank(),
      plot.margin = margin(10, 5, 10, 5),
      legend.position = "none"
    ) +
    labs(
      title = paste("Exp", exp_id, "(All fixation times combined)"),
      x = "False Positive Rate",
      y = "True Positive Rate"
    ) +
    scale_x_continuous(breaks = seq(0, 1, 0.2), limits = c(0, 1)) +
    scale_y_continuous(breaks = seq(0, 1, 0.2), limits = c(0, 1)) +
    coord_equal(xlim = c(0, 1), ylim = c(0, 1))
  
  auc_data <- data.frame()
  
  # One-vs-rest ROC for each tissue type
  for (tissue in names(tissue_colors)) {
    df_tissue <- df_sub %>%
      mutate(
        class = as.numeric(tissue_type == !!tissue),
        class = factor(class, levels = c("0", "1"))
      )
    
    # Skip if too few samples or no variation
    if (sum(df_tissue$class == 1) < 2 || sum(df_tissue$class == 0) < 2) next
    if (var(df_tissue$intensity) == 0) next
    
    roc_obj <- tryCatch(
      roc(response = df_tissue$class, predictor = df_tissue$intensity,
          quiet = TRUE, levels = c("0", "1"), direction = "<"),
      error = function(e) NULL
    )
    
    if (is.null(roc_obj)) next
    
    auc_val <- round(as.numeric(auc(roc_obj)), 3)
    star <- ifelse(auc_val > 0.8 | auc_val < 0.2, "*", "")
    
    auc_data <- rbind(auc_data, data.frame(
      tissue_type = tissue,
      AUC = auc_val,
      star = star,
      stringsAsFactors = FALSE
    ))
    
    coords_df <- coords(roc_obj, "all", ret = c("specificity", "sensitivity"), transpose = FALSE) %>%
      as.data.frame() %>%
      mutate(fpr = 1 - specificity)
    
    p <- p + geom_line(
      data = coords_df,
      aes(x = fpr, y = sensitivity),
      color = tissue_colors[tissue], size = 1.1, alpha = 0.9
    )
  }
  
  attr(p, "auc_data") <- auc_data
  return(p)
}

###main code 

base_dir <- "S:/266_and_267/Roc_PLOT_final/TT/ROC_PNGase_effect of TT/"
if (!dir.exists(base_dir)) dir.create(base_dir, recursive = TRUE)

mzs <- unique(df_long$mz)

cat("Starting export of", length(mzs), "m/z values (all fixation times combined)...\n")
pb <- txtProgressBar(min = 0, max = length(mzs), style = 3)

for (mzf in mzs) {
  setTxtProgressBar(pb, which(mzs == mzf))
  
  # Generate plots for both Experiments
  p266 <- plot_roc_one_exp(df_long, mzf, "266")
  p267 <- plot_roc_one_exp(df_long, mzf, "267")
  
  if (is.null(p266) && is.null(p267)) next
  
  blank <- ggplot() + theme_void() + ggtitle("No Data")
  p266 <- p266 %||% blank
  p267 <- p267 %||% blank
  
  auc_266 <- attr(p266, "auc_data") %||% data.frame()
  auc_267 <- attr(p267, "auc_data") %||% data.frame()
  
  legend_df <- bind_rows(
    auc_266 %>% mutate(exp = "266"),
    auc_267 %>% mutate(exp = "267")
  )
  
  # Build custom legend
  if (nrow(legend_df) == 0) {
    legend_plot <- ggplot() + theme_void() + ggtitle("No AUC")
  } else {
    legend_df <- legend_df %>%
      arrange(tissue_type, exp) %>%
      mutate(
        label = paste0("**", tissue_type, "** (Exp ", exp, ") AUC = ", AUC, star),
        y_pos = seq(0.95, by = -0.07, length.out = n())
      )
    
    swatch_grob <- segmentsGrob(
      x0 = unit(0.02, "npc"), y0 = unit(legend_df$y_pos, "npc"),
      x1 = unit(0.18, "npc"), y1 = unit(legend_df$y_pos, "npc"),
      gp = gpar(col = tissue_colors[legend_df$tissue_type], lwd = 4)
    )
    
    text_grob <- textGrob(
      label = legend_df$label,
      x = unit(0.22, "npc"),
      y = unit(legend_df$y_pos, "npc"),
      just = c("left", "centre"),
      gp = gpar(fontsize = 10, col = "black")
    )
    
    legend_grob <- grobTree(swatch_grob, text_grob)
    
    legend_plot <- ggplot() +
      annotation_custom(legend_grob) +
      xlim(0, 1) + ylim(0, 1) +
      theme_void() +
      ggtitle("Legend") +
      theme(plot.title = element_text(size = 10, face = "bold", hjust = 0.5))
  }
  
  # Combine plots (Exp 266, Exp 267)
  final_plot <- p266 + p267 + legend_plot +
    plot_layout(widths = c(4, 4, 2)) +
    plot_annotation(
      title = bquote(bold("ROC Curves: m/z =" ~ .(mzf) ~ " (All fixation times combined)")),
      theme = theme(plot.title = element_text(size = 13, hjust = 0.5, face = "bold"))
    )
  
  out_file <- file.path(base_dir, paste0("ROC_all_tissues_PNGase_base-on-tissuetype_", mzf, "_266vs267_combined.png"))
  ggsave(out_file, final_plot, width = 18, height = 8, dpi = 300, bg = "white")
}

close(pb)
