# AUC-ROC: ONE-VS-REST (m/z per fixation time) / m/z, all tissues combined) - 266 vs 267
# Combined across all tissue types!
# Repeated once for Trypsin (peptide dataset) and once for PNGase F (Glycan dataset)
# this code run for Glycan (PNGase F data set)


library(tidyverse)
library(pROC)
library(patchwork)
library(cowplot)
library(grid)

# plot colors for fixation groups
fixation_time_colors <- c(
  "3H"  = "#FF00FF",  # magenta
  "6H"  = "#FFFF00",  # yellow
  "12H" = "#0000FF",  # blue
  "24H" = "#FFA500",  # orange
  "60H" = "#00FF00"   # green
)

# load data
df_wide <- readRDS("S:/266_and_267/Roc_PLOT_final/FT/roc_PNGase_FT_effect/all_data_df_PNGase_combine_267and266.rds")

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
  filter(!is.na(intensity), !is.na(Experiment))


# Function : per sild and all tissue types combined 

plot_roc_one_exp_all_tissues <- function(df_long, mzf, exp_id) {
  
  df_sub <- df_long %>%
    filter(mz == !!as.character(mzf), Experiment == !!exp_id)
  
  if (nrow(df_sub) == 0) {
    warning(paste("No data for exp", exp_id))
    return(ggplot() + theme_void() + ggtitle(paste("Exp", exp_id, "- No Data")))
  }
  
  time_order <- c("3H", "6H", "12H", "24H", "60H")
  df_sub <- df_sub %>%
    mutate(fixation_time = factor(fixation_time, levels = time_order)) %>%
    filter(fixation_time %in% time_order)
  
  if (nrow(df_sub) == 0) {
    warning(paste("No valid fixation times for exp", exp_id))
    return(ggplot() + theme_void() + ggtitle(paste("Exp", exp_id, "- No Times")))
  }
  
  fix_times <- levels(df_sub$fixation_time)
  
  # Base ROC plot
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
      title = paste("Exp", exp_id, "- All tissues combined"),
      x = "False Positive Rate",
      y = "True Positive Rate"
    ) +
    scale_x_continuous(breaks = seq(0, 1, 0.2), limits = c(0, 1)) +
    scale_y_continuous(breaks = seq(0, 1, 0.2), limits = c(0, 1)) +
    coord_equal(xlim = c(0, 1), ylim = c(0, 1))
  
  auc_data <- data.frame()
  
  for (ft in fix_times) {
    df_class <- df_sub %>%
      filter(fixation_time %in% c(ft, setdiff(fix_times, ft))) %>%
      mutate(
        class = as.numeric(fixation_time == ft),
        class = factor(class, levels = c("0", "1"))
      )
    
    if (sum(df_class$class == 1) < 2 || sum(df_class$class == 0) < 2) next
    if (var(df_class$intensity) == 0) next
    
    roc_obj <- tryCatch(
      roc(response = df_class$class, predictor = df_class$intensity,
          quiet = TRUE, levels = c("0", "1"), direction = "<"),
      error = function(e) NULL
    )
    if (is.null(roc_obj)) next
    
    auc_val <- round(as.numeric(auc(roc_obj)), 3)
    star <- ifelse(auc_val > 0.8 | auc_val < 0.2, "*", "")
    
    auc_data <- rbind(auc_data, data.frame(
      fixation_time = ft, AUC = auc_val, star = star,
      stringsAsFactors = FALSE
    ))
    
    coords_df <- coords(roc_obj, "all", ret = c("specificity", "sensitivity"), transpose = FALSE) %>%
      as.data.frame() %>%
      mutate(fpr = 1 - specificity, fixation_time = ft)
    
    p <- p + geom_line(
      data = coords_df,
      aes(x = fpr, y = sensitivity),
      color = fixation_time_colors[ft], size = 1.1, alpha = 0.9
    )
  }
  
  attr(p, "auc_data") <- auc_data
  return(p)
}

#Checking:

str(df_long)
summary(df_long)

# 2 plots (slide 266 and 267) + legend

mzf <- "917.031"
exp_ids <- c("266", "267")

plot_list <- lapply(exp_ids, function(exp) {
  plot_roc_one_exp_all_tissues(df_long, mzf, exp)
})

auc_266 <- attr(plot_list[[1]], "auc_data")
auc_267 <- attr(plot_list[[2]], "auc_data")

legend_df <- bind_rows(
  auc_266 %>% mutate(exp = "266"),
  auc_267 %>% mutate(exp = "267")
) %>%
  arrange(factor(fixation_time, levels = c("3H", "6H", "12H", "24H", "60H")), exp) %>%
  mutate(
    label = paste0("**", fixation_time, "** (Exp ", exp, ")  AUC = ", AUC, star),
    y_pos = seq(0.95, by = -0.06, length.out = n())
  )

swatch_grob <- segmentsGrob(
  x0 = unit(0.02, "npc"), y0 = unit(legend_df$y_pos, "npc"),
  x1 = unit(0.18, "npc"), y1 = unit(legend_df$y_pos, "npc"),
  gp = gpar(col = fixation_time_colors[legend_df$fixation_time], lwd = 4)
)

text_grob <- textGrob(
  label = legend_df$label,
  x = unit(0.22, "npc"),
  y = unit(legend_df$y_pos, "npc"),
  just = c("left", "centre"),
  gp = gpar(fontsize = 10, col = "black")
)

ref_text <- paste0(
  "*AUC > 0.8: strong separation \n(higher intensity)\n",
  "*AUC < 0.2: strong reverse separation \n(lower intensity)"
)
ref_grob <- textGrob(
  label = ref_text,
  x = unit(0.02, "npc"), y = unit(0.08, "npc"),
  just = c("left", "bottom"),
  gp = gpar(fontsize = 8.5, col = "gray40", fontface = "italic")
)

legend_grob <- grobTree(swatch_grob, text_grob, ref_grob)

legend_plot <- ggplot() +
  annotation_custom(legend_grob) +
  xlim(0, 1) + ylim(0, 1) +
  theme_void() +
  ggtitle("Legend") +
  theme(plot.title = element_text(size = 11, face = "bold", hjust = 0.5, margin = margin(b = 10)))

final_plot <- plot_list[[1]] + plot_list[[2]] + legend_plot +
  plot_layout(widths = c(3.5, 3.5, 1.8)) +
  plot_annotation(
    title = bquote(bold("ROC Curves:") ~ m/z == .(mzf) ~ "| All Tissues Combined"),
    theme = theme(plot.title = element_text(size = 14, hjust = 0.5, face = "bold"))
  )

out_dir <- "H:/roc_auc/PNGase/results_all_tissues"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

out_file <- file.path(out_dir, paste0("ROC_PRO_allTissues_", mzf, "_266vs267.png"))
ggsave(out_file, final_plot, width = 14, height = 7, dpi = 300, bg = "white")
cat("Combined ROC plot (all tissues, 266 vs 267) saved:\n", out_file, "\n")


# code: ONE image PER m/z (ALL tissues combined)

base_dir <- "S:/266_and_267/Roc_PLOT_final/FT/roc_PNGase_FT_effect/plot"
if (!dir.exists(base_dir)) dir.create(base_dir, recursive = TRUE)

combinations <- df_long %>%
  distinct(mz) %>%
  arrange(mz)

cat("Starting export of", nrow(combinations), "m/z combinations (all tissues combined)...\n")
pb <- txtProgressBar(min = 0, max = nrow(combinations), style = 3)

for (i in seq_len(nrow(combinations))) {
  mzf <- combinations$mz[i]
  setTxtProgressBar(pb, i)
  
  p266 <- plot_roc_one_exp_all_tissues(df_long, mzf, "266")
  p267 <- plot_roc_one_exp_all_tissues(df_long, mzf, "267")
  
  if (is.null(p266) && is.null(p267)) next
  
  blank <- ggplot() + theme_void() + ggtitle("No Data")
  p266 <- p266 %||% blank + ggtitle("Exp 266")
  p267 <- p267 %||% blank + ggtitle("Exp 267")
  
  auc_266 <- attr(p266, "auc_data") %||% data.frame()
  auc_267 <- attr(p267, "auc_data") %||% data.frame()
  
  legend_df <- bind_rows(
    auc_266 %>% mutate(exp = "266"),
    auc_267 %>% mutate(exp = "267")
  )
  
  if (nrow(legend_df) == 0) {
    legend_plot <- ggplot() + theme_void() + ggtitle("No AUC")
  } else {
    legend_df <- legend_df %>%
      arrange(factor(fixation_time, levels = c("3H","6H","12H","24H","60H")), exp) %>%
      mutate(
        label = paste0("**", fixation_time, "** (Exp ", exp, ") AUC = ", AUC, star),
        y_pos = seq(0.95, by = -0.07, length.out = n())
      )
    
    swatch_grob <- segmentsGrob(
      x0 = unit(0.02, "npc"), y0 = unit(legend_df$y_pos, "npc"),
      x1 = unit(0.18, "npc"), y1 = unit(legend_df$y_pos, "npc"),
      gp = gpar(col = fixation_time_colors[legend_df$fixation_time], lwd = 4)
    )
    
    text_grob <- textGrob(
      label = legend_df$label,
      x = unit(0.22, "npc"),
      y = unit(legend_df$y_pos, "npc"),
      just = c("left", "centre"),
      gp = gpar(fontsize = 10, col = "black")
    )
    
    ref_text <- paste0(
      "*AUC > 0.8: strong separation \n(higher intensity)\n",
      "*AUC < 0.2: strong reverse separation \n(lower intensity)"
    )
    ref_grob <- textGrob(
      label = ref_text,
      x = unit(0.02, "npc"), y = unit(0.08, "npc"),
      just = c("left", "bottom"),
      gp = gpar(fontsize = 8.5, col = "gray40", fontface = "italic")
    )
    
    legend_grob <- grobTree(swatch_grob, text_grob, ref_grob)
    
    legend_plot <- ggplot() +
      annotation_custom(legend_grob) +
      xlim(0, 1) + ylim(0, 1) +
      theme_void() +
      ggtitle("Legend") +
      theme(plot.title = element_text(size = 10, face = "bold", hjust = 0.5))
  }
  
  final_plot <- p266 + p267 + legend_plot +
    plot_layout(widths = c(4, 4, 2)) +
    plot_annotation(
      title = bquote(bold("All Tissues Combined") ~ "|" ~ m/z == .(mzf)),
      theme = theme(plot.title = element_text(size = 13, hjust = 0.5, face = "bold"))
    )
  
  out_file <- file.path(base_dir, paste0("ROC_PNGase_allTissues_", mzf, "_266vs267.png"))
  ggsave(out_file, final_plot, width = 15, height = 6.5, dpi = 300, bg = "white")
}

close(pb)

