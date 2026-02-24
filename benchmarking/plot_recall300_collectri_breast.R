#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
input_csv <- if (length(args) >= 1) args[[1]] else "benchmark_results_20260216_203141/benchmark.csv"
output_png <- if (length(args) >= 2) args[[2]] else sub("\\.csv$", "_recall300_collectri_breast.png", input_csv)

required_packages <- c("ggplot2", "dplyr")
missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages) > 0) {
  stop(
    sprintf(
      "Missing required package(s): %s. Install with: install.packages(c(%s))",
      paste(missing_packages, collapse = ", "),
      paste(sprintf('"%s"', missing_packages), collapse = ", ")
    )
  )
}

library(dplyr)

if (!file.exists(input_csv)) {
  stop(sprintf("Input file not found: %s", input_csv))
}

df <- read.csv(input_csv, check.names = FALSE, stringsAsFactors = FALSE)

required_cols <- c("Method", "Recall@300", "Network", "Disease")
missing_cols <- setdiff(required_cols, names(df))
if (length(missing_cols) > 0) {
  stop(sprintf("Input CSV missing required columns: %s", paste(missing_cols, collapse = ", ")))
}

df$`Recall@300` <- as.numeric(df$`Recall@300`)
df$Network <- as.character(df$Network)
df$Method <- as.character(df$Method)
df$Disease <- as.character(df$Disease)

df <- df[!is.na(df$`Recall@300`), ]

# Filter: CollecTRI (all) OR (WL + breast neoplasms)
df_filtered <- df %>%
  dplyr::filter(
    (Network == "collectri") |
    (Network == "wl" & Disease == "breast neoplasms")
  )

# Aggregate: mean Recall@300 per disease per method per network
df_agg <- df_filtered %>%
  dplyr::group_by(Method, Network, Disease) %>%
  dplyr::summarise(
    `Recall@300` = mean(`Recall@300`, na.rm = TRUE),
    .groups = "drop"
  )

message(sprintf("Filtered to CollecTRI (all) + WL breast neoplasms"))
message(sprintf("CollecTRI diseases: %d", n_distinct(df_agg$Disease[df_agg$Network == "collectri"])))
message(sprintf("WL (breast neoplasms) diseases: %d", n_distinct(df_agg$Disease[df_agg$Network == "wl"])))

# Prepare plot data for each network
collectri_data <- df_agg %>% dplyr::filter(Network == "collectri")
wl_data <- df_agg %>% dplyr::filter(Network == "wl")

# Combine data and create single plot
combined_data <- rbind(collectri_data, wl_data)

# Compute mean Recall@300 per method-network combination for bars
bar_data <- combined_data %>%
  dplyr::group_by(Method, Network) %>%
  dplyr::summarise(
    mean_recall = mean(`Recall@300`, na.rm = TRUE),
    .groups = "drop"
  )

# Create a combined plot with bars (geom_col) grouped by network, points colored by disease
p <- ggplot2::ggplot() +
  ggplot2::geom_col(
    data = bar_data,
    ggplot2::aes(x = reorder(Method, mean_recall, FUN = median), y = mean_recall, fill = Network),
    alpha = 0.7,
    position = ggplot2::position_dodge(width = 0.75),
    width = 0.6
  ) +
  ggplot2::geom_point(
    data = combined_data,
    ggplot2::aes(x = reorder(Method, `Recall@300`, FUN = median), y = `Recall@300`,
                 group = Network, color = Disease),
    alpha = 0.6,
    size = 2,
    position = ggplot2::position_dodge(width = 0.75)
  ) +
  ggplot2::scale_fill_manual(values = c("collectri" = "steelblue", "wl" = "coral")) +
  ggplot2::theme_minimal() +
  ggplot2::theme(
    axis.text.x = ggplot2::element_text(angle = 45, hjust = 1, vjust = 1, size = 10),
    axis.text.y = ggplot2::element_text(size = 10),
    axis.title.x = ggplot2::element_text(size = 12, face = "bold"),
    axis.title.y = ggplot2::element_text(size = 12, face = "bold"),
    legend.text = ggplot2::element_text(size = 9),
    legend.title = ggplot2::element_text(size = 10, face = "bold"),
    legend.position = "right",
    panel.grid.major.y = ggplot2::element_line(color = "gray90"),
    panel.grid.minor.y = ggplot2::element_blank(),
    panel.grid.major.x = ggplot2::element_blank()
  ) +
  ggplot2::labs(
    title = "Recall@300 Distribution by Method and Network",
    subtitle = "Boxplots show CollecTRI (4 diseases) vs WL breast neoplasms; points colored by disease",
    x = "Walking Method",
    y = "Recall@300 (Fraction of Test Genes Recovered)",
    fill = "Network",
    color = "Disease"
  )

width <- 16
height <- 6
ggplot2::ggsave(
  filename = output_png,
  plot = p,
  width = width,
  height = height,
  dpi = 300,
  bg = "white"
)

message(sprintf("Saved side-by-side boxplot to: %s", output_png))
message(sprintf("Plot dimensions: %.1f x %.1f inches (dpi 300)", width, height))
