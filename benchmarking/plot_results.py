"""Generate plots from benchmark results.

This script creates visualizations from the unified benchmark output.
Reads from benchmark_plotting.csv and benchmark_method_comparison.csv.

Usage:
    python benchmarking/plot_results.py
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Set style for better-looking plots
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 300


def plot_time_vs_performance(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot execution time vs various performance metrics."""
    metrics = ["Accuracy", "AP", "Genes_obtained_300"]
    metric_labels = ["Accuracy (AUROC)", "AP", "Recall@300"]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes = axes.flatten()
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        sns.scatterplot(
            data=df,
            x="Time_s",
            y=metric,
            hue="Method",
            style="Network",
            s=100,
            ax=axes[i],
            alpha=0.7,
        )
        axes[i].set_xlabel("Execution Time (s)", fontsize=12)
        axes[i].set_ylabel(label, fontsize=12)
        axes[i].set_title(f"Time vs {label}", fontsize=13, fontweight="bold")
        axes[i].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / "time_vs_performance.png", dpi=300, bbox_inches="tight")
    logger.info("✓ Saved time_vs_performance.png")
    plt.close()


def plot_method_comparison(comparison_df: pd.DataFrame, output_dir: Path) -> None:
    """Create bar charts comparing methods across key metrics."""
    # Sort by Recall@300 for better visualization
    comparison_df = comparison_df.sort_values("Recall@300_mean", ascending=True)
    
    # Check if MRR is available
    has_mrr = "MRR@300_mean" in comparison_df.columns
    n_plots = 4 if has_mrr else 3
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 6))
    
    plot_idx = 0
    
    # Plot 1: Recall@300
    axes[plot_idx].barh(comparison_df["Method"], comparison_df["Recall@300_mean"], color="steelblue")
    axes[plot_idx].set_xlabel("Recall@300", fontsize=12)
    axes[plot_idx].set_title("Gene Recovery (Recall@300)", fontsize=13, fontweight="bold")
    axes[plot_idx].grid(axis="x", alpha=0.3)
    plot_idx += 1
    
    # Plot 2: MRR (if available)
    if has_mrr:
        mrr_sorted = comparison_df.sort_values("MRR@300_mean", ascending=True)
        axes[plot_idx].barh(mrr_sorted["Method"], mrr_sorted["MRR@300_mean"], color="coral")
        axes[plot_idx].set_xlabel("MRR@300", fontsize=12)
        axes[plot_idx].set_title("Mean Reciprocal Rank @300", fontsize=13, fontweight="bold")
        axes[plot_idx].grid(axis="x", alpha=0.3)
        plot_idx += 1
    
    # Plot 3: AP
    axes[plot_idx].barh(comparison_df["Method"], comparison_df["AP_mean"], color="mediumseagreen")
    axes[plot_idx].set_xlabel("AP", fontsize=12)
    axes[plot_idx].set_title("Average Precision", fontsize=13, fontweight="bold")
    axes[plot_idx].grid(axis="x", alpha=0.3)
    plot_idx += 1
    
    # Plot 4: Time
    comparison_time = comparison_df.sort_values("Time_mean", ascending=True)
    axes[plot_idx].barh(comparison_time["Method"], comparison_time["Time_mean"], color="orange")
    axes[plot_idx].set_xlabel("Execution Time (s)", fontsize=12)
    axes[plot_idx].set_title("Computational Speed", fontsize=13, fontweight="bold")
    axes[plot_idx].grid(axis="x", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "method_comparison.png", dpi=300, bbox_inches="tight")
    logger.info("✓ Saved method_comparison.png")
    plt.close()


def plot_recall_progression(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot how recall improves from @25 to @100 to @300."""
    # Reshape data for plotting
    recall_data = []
    for _, row in df.iterrows():
        for k, recall in [(25, "Genes_obtained_25"), (100, "Genes_obtained_100"), (300, "Genes_obtained_300")]:
            recall_data.append({
                "Method": row["Method"],
                "Network": row["Network"],
                "k": k,
                "Recall": row[recall],
            })
    recall_df = pd.DataFrame(recall_data)
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=recall_df,
        x="k",
        y="Recall",
        hue="Method",
        style="Network",
        markers=True,
        dashes=False,
        markersize=8,
        linewidth=2,
    )
    plt.xlabel("Top-k Predictions", fontsize=12)
    plt.ylabel("Recall", fontsize=12)
    plt.title("Gene Recovery at Different Cutoffs", fontsize=14, fontweight="bold")
    plt.xticks([25, 100, 300])
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "recall_progression.png", dpi=300, bbox_inches="tight")
    logger.info("✓ Saved recall_progression.png")
    plt.close()


def plot_mrr_vs_ap(df: pd.DataFrame, mrr_df: pd.DataFrame, output_dir: Path) -> None:
    """Scatter plot of MRR vs AP to show correlation."""
    # Merge MRR data with main data
    merged = df.groupby("Method", as_index=False).agg({"AP": "mean"})
    merged = merged.merge(mrr_df[["Method", "MRR@300_mean"]], on="Method", how="inner")
    
    if len(merged) == 0:
        logger.warning("No MRR data available for MRR vs AP plot")
        return
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=merged,
        x="MRR@300_mean",
        y="AP",
        s=150,
        alpha=0.7,
    )
    # Add method labels
    for _, row in merged.iterrows():
        plt.annotate(
            row["Method"],
            (row["MRR@300_mean"], row["AP"]),
            fontsize=8,
            alpha=0.7,
        )
    plt.xlabel("Mean Reciprocal Rank @300 (MRR)", fontsize=12)
    plt.ylabel("Average Precision (AP)", fontsize=12)
    plt.title("MRR@300 vs AP Correlation", fontsize=14, fontweight="bold")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "mrr_vs_ap.png", dpi=300, bbox_inches="tight")
    logger.info("✓ Saved mrr_vs_ap.png")
    plt.close()


def plot_network_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """Compare method performance across different networks."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    metrics = ["Genes_obtained_300", "MRR", "AP"]
    titles = ["Recall@300 by Network", "MRR by Network", "AP by Network"]
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        pivot_data = df.pivot_table(
            index="Method",
            columns="Network",
            values=metric,
            aggfunc="mean",
        )
        pivot_data.plot(kind="barh", ax=axes[i], width=0.7)
        axes[i].set_title(title, fontsize=13, fontweight="bold")
        axes[i].set_xlabel(metric.replace("_", " ").replace("Genes obtained", "Recall@"), fontsize=11)
        axes[i].legend(title="Network", fontsize=9)
        axes[i].grid(axis="x", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "network_comparison.png", dpi=300, bbox_inches="tight")
    logger.info("✓ Saved network_comparison.png")
    plt.close()


def main() -> None:
    """Generate all plots from benchmark results."""
    input_dir = Path("benchmarking")
    output_dir = Path("benchmarking") / "plots"
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    plotting_file = input_dir / "benchmark_plotting.csv"
    comparison_file = input_dir / "benchmark_method_comparison.csv"
    mrr300_file = input_dir / "benchmark_mrr300.csv"
    
    if not plotting_file.exists():
        logger.error("Error: %s not found. Run benchmark.py first.", plotting_file)
        return
    
    if not comparison_file.exists():
        logger.error("Error: %s not found. Run benchmark.py first.", comparison_file)
        return
    
    df = pd.read_csv(plotting_file)
    comparison_df = pd.read_csv(comparison_file)
    
    # Load MRR if available
    mrr_df = None
    if mrr300_file.exists():
        mrr_df = pd.read_csv(mrr300_file)
        logger.info("Loaded MRR@300 data: %d methods", len(mrr_df))
    
    logger.info("Loaded data: %d rows from plotting table, %d methods", len(df), len(comparison_df))
    logger.info("Generating plots...")
    
    # Generate all plots
    plot_time_vs_performance(df, output_dir)
    plot_method_comparison(comparison_df, output_dir)
    plot_recall_progression(df, output_dir)
    if mrr_df is not None and len(mrr_df) > 0:
        plot_mrr_vs_ap(df, mrr_df, output_dir)
    plot_network_comparison(df, output_dir)
    
    logger.info("\n" + "="*60)
    logger.info("✓ All plots saved to %s", output_dir)
    logger.info("="*60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    main()
