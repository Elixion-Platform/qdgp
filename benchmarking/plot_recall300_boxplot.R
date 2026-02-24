#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
input_csv <- if (length(args) >= 1) args[[1]] else "benchmark_results_20260216_203141/benchmark.csv"
output_png <- if (length(args) >= 2) args[[2]] else sub("\\.csv$", "_recall300_boxplot.png", input_csv)

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

# Aggregate: mean Recall@300 per disease per method per network
df_agg <- df %>%
  dplyr::group_by(Method, Network, Disease) %>%
  dplyr::summarise(
    `Recall@300` = mean(`Recall@300`, na.rm = TRUE),
    .groups = "drop"
  )

df <- df_agg
message(sprintf("Aggregated %d runs into %d disease-level means", nrow(df_agg), nrow(df)))

networks_present <- sort(unique(df$Network))
message(sprintf("Networks found: %s", paste(networks_present, collapse = ", ")))

p <- ggplot2::ggplot(
  df,
  ggplot2::aes(x = reorder(Method, `Recall@300`, FUN = median), y = `Recall@300`, fill = Method)
) +
  ggplot2::geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  ggplot2::geom_jitter(width = 0.2, height = 0, alpha = 0.4, size = 2, color = "black") +
  ggplot2::facet_wrap(~Network, nrow = 1, scales = "free_x") +
  ggplot2::theme_minimal() +
  ggplot2::theme(
    axis.text.x = ggplot2::element_text(angle = 45, hjust = 1, vjust = 1, size = 10),
    axis.text.y = ggplot2::element_text(size = 10),
    axis.title.x = ggplot2::element_text(size = 12, face = "bold"),
    axis.title.y = ggplot2::element_text(size = 12, face = "bold"),
    strip.text = ggplot2::element_text(size = 11, face = "bold"),
    legend.position = "none",
    panel.grid.major.y = ggplot2::element_line(color = "gray90"),
    panel.grid.minor.y = ggplot2::element_blank(),
    panel.grid.major.x = ggplot2::element_blank(),
    figure.width = 16,
    figure.height = 6
  ) +
  ggplot2::labs(
    title = "Recall@300 Distribution by Method and Network",
    subtitle = "Each point represents the mean Recall@300 for one disease (averaged across 10 runs); boxplot shows median and quartiles across diseases",
    x = "Walking Method",
    y = "Recall@300 (Fraction of Test Genes Recovered)"
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

message(sprintf("Saved Recall@300 boxplot to: %s", output_png))
message(sprintf("Plot dimensions: %.1f x %.1f inches (dpi 300)", width, height))
