"""Enhanced benchmark processing with better visualizations and analysis."""
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

OUTPUT_DIR = Path("benchmarking")


def parse_method_info(method_name: str) -> Dict[str, str]:
    """Extract structured information from method names.
    
    Returns dictionary with category, algorithm, and parameters.
    """
    info = {
        "Method": method_name,
        "Category": "Unknown",
        "Algorithm": method_name,
        "Walk_Type": "Unknown",
    }
    
    if method_name.startswith("qwalker_rw"):
        info["Category"] = "QWalker"
        info["Walk_Type"] = "Classical"
        info["Algorithm"] = "Random_Walk"
        # Parse parameters like rp0.15, steps50
        parts = method_name.replace("qwalker_rw_", "").split("_")
        for part in parts:
            if part.startswith("rp"):
                info["restart_prob"] = part[2:]
            elif part.startswith("steps"):
                info["n_steps"] = part[5:]
            elif part in ["matrix", "mc"]:
                info["mode"] = part
            elif part.startswith("n"):
                info["n_walkers"] = part[1:]
                
    elif method_name.startswith("qwalker_qw"):
        info["Category"] = "QWalker"
        info["Walk_Type"] = "Quantum"
        info["Algorithm"] = "Quantum_Walk"
        # Parse parameters like t0.5, adjacency
        parts = method_name.replace("qwalker_qw_", "").split("_")
        for part in parts:
            if part.startswith("t"):
                info["time"] = part[1:]
            elif part in ["adjacency", "laplacian"]:
                info["hamiltonian"] = part
                
    elif method_name.startswith("benchmark_"):
        info["Category"] = "Baseline"
        info["Walk_Type"] = "Baseline"
        algo = method_name.replace("benchmark_", "")
        info["Algorithm"] = algo.upper()
        
    return info


def enhance_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add parsed method information columns."""
    df = df.copy()
    
    # Parse all method info
    method_info = df["Method"].apply(parse_method_info)
    info_df = pd.DataFrame(method_info.tolist())
    
    # Add new columns
    for col in ["Category", "Algorithm", "Walk_Type"]:
        if col in info_df.columns:
            df[col] = info_df[col]
    
    return df


def create_summary_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create multiple summary views."""
    
    # Overall summary by method
    summary_method = (
        df.groupby(["Method", "Category", "Algorithm", "Walk_Type", "Network"], as_index=False)
        .agg({
            "Time (s)": ["mean", "std", "min", "max"],
            "Num_seeds": "mean",
            "Run": "count"
        })
    )
    summary_method.columns = [
        "Method", "Category", "Algorithm", "Walk_Type", "Network",
        "Time_mean", "Time_std", "Time_min", "Time_max",
        "Avg_seeds", "Runs"
    ]
    summary_method = summary_method.sort_values(["Network", "Time_mean"])
    
    # Summary by seed size
    summary_seeds = (
        df.groupby(["Method", "Num_seeds", "Network"], as_index=False)
        .agg({"Time (s)": ["mean", "std"], "Run": "count"})
    )
    summary_seeds.columns = [
        "Method", "Num_seeds", "Network",
        "Time_mean", "Time_std", "Runs"
    ]
    
    # Category comparison
    summary_category = (
        df.groupby(["Category", "Walk_Type", "Network"], as_index=False)
        .agg({
            "Time (s)": ["mean", "std", "min", "max"],
            "Method": "nunique"
        })
    )
    summary_category.columns = [
        "Category", "Walk_Type", "Network",
        "Time_mean", "Time_std", "Time_min", "Time_max",
        "Num_methods"
    ]
    
    return summary_method, summary_seeds, summary_category


def create_visualizations(df: pd.DataFrame, output_dir: Path):
    """Create comprehensive visualization suite."""
    
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    # 1. Overall comparison by method and network
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, network in enumerate(df["Network"].unique()):
        network_data = df[df["Network"] == network]
        ax = axes[idx] if len(df["Network"].unique()) > 1 else axes
        
        # Box plot for each method
        sns.boxplot(
            data=network_data,
            y="Method",
            x="Time (s)",
            ax=ax,
            hue="Category"
        )
        ax.set_title(f"Algorithm Performance - {network.upper()}")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Method")
    
    plt.tight_layout()
    plt.savefig(output_dir / "benchmark_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 2. Performance by seed size
    g = sns.relplot(
        data=df,
        x="Num_seeds",
        y="Time (s)",
        hue="Method",
        col="Network",
        kind="line",
        height=5,
        aspect=1.2,
        markers=True,
        style="Category"
    )
    g.set_titles("{col_name} Network")
    g.set_axis_labels("Number of Seeds", "Time (seconds)")
    plt.savefig(output_dir / "benchmark_by_seeds.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 3. Category comparison
    if "Category" in df.columns:
        fig, axes = plt.subplots(1, len(df["Network"].unique()), figsize=(14, 6))
        if len(df["Network"].unique()) == 1:
            axes = [axes]
        
        for idx, network in enumerate(df["Network"].unique()):
            network_data = df[df["Network"] == network]
            ax = axes[idx]
            
            sns.violinplot(
                data=network_data,
                x="Category",
                y="Time (s)",
                ax=ax,
                hue="Walk_Type"
            )
            ax.set_title(f"{network.upper()}")
            ax.set_xlabel("Method Category")
            ax.set_ylabel("Time (seconds)")
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / "benchmark_by_category.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    # 4. Heatmap of average performance
    pivot_data = df.groupby(["Method", "Network"])["Time (s)"].mean().reset_index()
    pivot_table = pivot_data.pivot(index="Method", columns="Network", values="Time (s)")
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        cbar_kws={"label": "Average Time (s)"}
    )
    plt.title("Average Performance Heatmap")
    plt.xlabel("Network")
    plt.ylabel("Method")
    plt.tight_layout()
    plt.savefig(output_dir / "benchmark_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 5. QWalker-specific analysis (if available)
    qwalker_data = df[df["Category"] == "QWalker"]
    if not qwalker_data.empty:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Classical vs Quantum
        sns.boxplot(
            data=qwalker_data,
            x="Walk_Type",
            y="Time (s)",
            hue="Network",
            ax=axes[0]
        )
        axes[0].set_title("QWalker: Classical vs Quantum Performance")
        axes[0].set_xlabel("Walk Type")
        axes[0].set_ylabel("Time (seconds)")
        
        # Performance distribution
        sns.stripplot(
            data=qwalker_data,
            x="Walk_Type",
            y="Time (s)",
            hue="Network",
            dodge=True,
            alpha=0.6,
            size=4,
            ax=axes[1]
        )
        axes[1].set_title("QWalker: Performance Distribution")
        axes[1].set_xlabel("Walk Type")
        axes[1].set_ylabel("Time (seconds)")
        
        plt.tight_layout()
        plt.savefig(output_dir / "qwalker_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()


def generate_report(
    df: pd.DataFrame,
    summary_method: pd.DataFrame,
    summary_category: pd.DataFrame,
    output_dir: Path
):
    """Generate a text report with key findings."""
    
    report = []
    report.append("=" * 80)
    report.append("BENCHMARK ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Overall statistics
    report.append("OVERALL STATISTICS:")
    report.append(f"  Total runs: {len(df)}")
    report.append(f"  Unique methods: {df['Method'].nunique()}")
    report.append(f"  Networks tested: {', '.join(df['Network'].unique())}")
    report.append(f"  Diseases/conditions: {df['Disease'].nunique()}")
    report.append(f"  Seed sizes range: {df['Num_seeds'].min()} - {df['Num_seeds'].max()}")
    report.append("")
    
    # Best performers by network
    report.append("FASTEST METHODS BY NETWORK:")
    for network in df["Network"].unique():
        network_summary = summary_method[summary_method["Network"] == network].nsmallest(3, "Time_mean")
        report.append(f"\n  {network.upper()}:")
        for _, row in network_summary.iterrows():
            report.append(f"    {row['Method']}: {row['Time_mean']:.4f}s (±{row['Time_std']:.4f})")
    report.append("")
    
    # Category comparison
    if not summary_category.empty:
        report.append("PERFORMANCE BY CATEGORY:")
        for _, row in summary_category.iterrows():
            report.append(
                f"  {row['Category']} ({row['Walk_Type']}) on {row['Network']}: "
                f"{row['Time_mean']:.4f}s (±{row['Time_std']:.4f})"
            )
    report.append("")
    
    # QWalker analysis
    qwalker_data = df[df["Category"] == "QWalker"]
    if not qwalker_data.empty:
        report.append("QWALKER ANALYSIS:")
        for walk_type in qwalker_data["Walk_Type"].unique():
            type_data = qwalker_data[qwalker_data["Walk_Type"] == walk_type]
            mean_time = type_data["Time (s)"].mean()
            std_time = type_data["Time (s)"].std()
            report.append(f"  {walk_type}: {mean_time:.4f}s (±{std_time:.4f})")
        report.append("")
        
        # Compare to baseline
        baseline_mean = df[df["Category"] == "Baseline"]["Time (s)"].mean()
        qwalker_mean = qwalker_data["Time (s)"].mean()
        ratio = qwalker_mean / baseline_mean if baseline_mean > 0 else float('inf')
        report.append(f"  QWalker avg vs Baseline avg: {ratio:.2f}x")
        if ratio > 1:
            report.append(f"  → QWalker is {ratio:.2f}x slower on average")
        else:
            report.append(f"  → QWalker is {1/ratio:.2f}x faster on average")
    
    report.append("")
    report.append("=" * 80)
    
    report_text = "\n".join(report)
    print(report_text)
    
    with open(output_dir / "benchmark_report.txt", "w") as f:
        f.write(report_text)


def main():
    """Main processing function."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Read raw results
    benchmark_file = OUTPUT_DIR / "benchmark.csv"
    if not benchmark_file.exists():
        print(f"❌ Benchmark file not found: {benchmark_file}")
        print("   Run the benchmark first: python benchmarking/benchmark.py")
        return
    
    print("Loading benchmark results...")
    res_df = pd.read_csv(benchmark_file)
    
    # Enhance with parsed information
    print("Enhancing data with method categories...")
    res_df = enhance_dataframe(res_df)
    
    # Create summaries
    print("Creating summary tables...")
    summary_method, summary_seeds, summary_category = create_summary_tables(res_df)
    
    # Save enhanced data and summaries
    res_df.to_csv(OUTPUT_DIR / "benchmark_enhanced.csv", index=False)
    summary_method.to_csv(OUTPUT_DIR / "summary_by_method.csv", index=False)
    summary_seeds.to_csv(OUTPUT_DIR / "summary_by_seeds.csv", index=False)
    summary_category.to_csv(OUTPUT_DIR / "summary_by_category.csv", index=False)
    
    print(f"✓ Saved enhanced CSV files to {OUTPUT_DIR}/")
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(res_df, OUTPUT_DIR)
    print(f"✓ Saved plots to {OUTPUT_DIR}/")
    
    # Generate report
    print("\nGenerating report...")
    generate_report(res_df, summary_method, summary_category, OUTPUT_DIR)
    
    print("\n✅ Benchmark processing complete!")
    print(f"\nOutput files in {OUTPUT_DIR}:")
    print("  - benchmark_enhanced.csv (raw data with categories)")
    print("  - summary_by_method.csv (aggregated by method)")
    print("  - summary_by_seeds.csv (aggregated by seed size)")
    print("  - summary_by_category.csv (aggregated by category)")
    print("  - benchmark_comparison.png (overall comparison)")
    print("  - benchmark_by_seeds.png (performance vs seed size)")
    print("  - benchmark_by_category.png (category comparison)")
    print("  - benchmark_heatmap.png (performance heatmap)")
    print("  - qwalker_analysis.png (QWalker-specific analysis)")
    print("  - benchmark_report.txt (text summary)")


if __name__ == "__main__":
    main()
