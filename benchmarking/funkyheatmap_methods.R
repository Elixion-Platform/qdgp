#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
input_csv <- if (length(args) >= 1) args[[1]] else "benchmark_results_20260216_203141/benchmark_method_comparison.csv"
output_png <- if (length(args) >= 2) args[[2]] else sub("\\.csv$", "_funkyheatmap.png", input_csv)
size_scale <- if (length(args) >= 3) as.numeric(args[[3]]) else 1.8

required_packages <- c("funkyheatmap", "ggplot2")
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

if (!file.exists(input_csv)) {
  stop(sprintf("Input file not found: %s", input_csv))
}

df <- read.csv(input_csv, check.names = FALSE, stringsAsFactors = FALSE)

if (!("Method" %in% names(df))) {
  stop("Input CSV must contain a 'Method' column.")
}

mean_cols <- grep("_mean$", names(df), value = TRUE)
mean_cols <- setdiff(mean_cols, c("Avg_Num_seeds", "Num_seeds_mean"))
ranking_cols <- mean_cols[!grepl("^Time", mean_cols)]

if (length(mean_cols) == 0) {
  stop("No metric columns ending with '_mean' were found.")
}

if (length(ranking_cols) == 0) {
  stop("No non-time metric columns found for ranking.")
}

normalize_01 <- function(x) {
  x <- as.numeric(x)
  rng <- range(x, na.rm = TRUE)
  if (!is.finite(rng[1]) || !is.finite(rng[2])) return(rep(NA_real_, length(x)))
  if (rng[1] == rng[2]) return(rep(0.5, length(x)))
  (x - rng[1]) / (rng[2] - rng[1])
}

score_matrix <- sapply(ranking_cols, function(col) {
  x <- as.numeric(df[[col]])
  normalize_01(x)
})

if (is.null(dim(score_matrix))) {
  score_matrix <- matrix(score_matrix, ncol = 1)
  colnames(score_matrix) <- ranking_cols
}

df$CompositeScore <- rowMeans(score_matrix, na.rm = TRUE)
df <- df[order(df$CompositeScore, decreasing = TRUE), , drop = FALSE]

to_pretty_name <- function(x) {
  x <- sub("_mean$", "", x)
  x <- gsub("True_Hits@", "Hits@", x)
  x <- gsub("AUROC", "AUROC", x)
  x <- gsub("AP", "AP", x)
  x <- gsub("MRR@", "MRR@", x)
  x
}

plot_data <- data.frame(
  id = df$Method,
  Method = df$Method,
  stringsAsFactors = FALSE
)

for (col in mean_cols) {
  plot_data[[col]] <- as.numeric(df[[col]])
  plot_data[[paste0(col, "_txt")]] <- sprintf("%.3f", as.numeric(df[[col]]))
}

column_info <- data.frame(
  id = character(),
  name = character(),
  geom = character(),
  group = character(),
  palette = character(),
  overlay = logical(),
  legend = logical(),
  size = numeric(),
  stringsAsFactors = FALSE
)

# Explicit method labels as first visible column (keep raw method IDs unchanged)
column_info <- rbind(
  column_info,
  data.frame(
    id = "Method",
    name = "Method",
    geom = "text",
    group = "Method",
    palette = NA_character_,
    overlay = FALSE,
    legend = FALSE,
    size = 3.2,
    stringsAsFactors = FALSE
  )
)

for (col in mean_cols) {
  group_name <- if (grepl("^Time", col)) "Efficiency" else "Accuracy"
  palette_name <- if (grepl("^Time", col)) "efficiency_palette" else "accuracy_palette"

  column_info <- rbind(
    column_info,
    data.frame(
      id = col,
      name = to_pretty_name(col),
      geom = "funkyrect",
      group = group_name,
      palette = palette_name,
      overlay = FALSE,
      legend = FALSE,
      size = NA_real_,
      stringsAsFactors = FALSE
    ),
    data.frame(
      id = paste0(col, "_txt"),
      name = "",
      geom = "text",
      group = group_name,
      palette = NA_character_,
      overlay = TRUE,
      legend = FALSE,
      size = 2.8,
      stringsAsFactors = FALSE
    )
  )
}

row_info <- data.frame(
  id = plot_data$id,
  group = NA_character_,
  stringsAsFactors = FALSE
)

column_groups <- data.frame(
  group = c("Method", "Accuracy", "Efficiency"),
  level1 = c("Method", "Accuracy Metrics", "Efficiency Metric"),
  stringsAsFactors = FALSE
)

palettes <- list(
  accuracy_palette = "Blues",
  efficiency_palette = "Reds"
)

legends <- list(
  list(palette = "accuracy_palette", enabled = FALSE),
  list(palette = "efficiency_palette", enabled = FALSE)
)

heat <- funkyheatmap::funky_heatmap(
  data = plot_data,
  column_info = column_info,
  row_info = row_info,
  column_groups = column_groups,
  palettes = palettes,
  legends = legends,
  scale_column = TRUE,
  add_abc = FALSE
)

ggplot2::ggsave(
  filename = output_png,
  plot = heat,
  width = max(12, heat$width * size_scale),
  height = max(8, heat$height * size_scale),
  dpi = 300,
  bg = "white"
)

message(sprintf("Saved funky heatmap to: %s", output_png))
message("Methods are ordered by CompositeScore (mean of min-max normalized non-time metric means); time is shown in the plot but excluded from ranking.")
