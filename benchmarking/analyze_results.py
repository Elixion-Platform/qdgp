"""Example analysis script using benchmark results.

Demonstrates how to load and analyze the unified benchmark data.
"""

import pandas as pd


def load_data():
    """Load benchmark data from CSV files."""
    plotting_df = pd.read_csv("benchmarking/benchmark_plotting.csv")
    comparison_df = pd.read_csv("benchmarking/benchmark_method_comparison.csv")
    
    # Load MRR if available
    try:
        mrr300_df = pd.read_csv("benchmarking/benchmark_mrr300.csv")
        # Merge MRR into comparison
        comparison_df = comparison_df.merge(mrr300_df, on="Method", how="left")
    except FileNotFoundError:
        pass
    
    return plotting_df, comparison_df


def analyze_performance():
    """Example analyses you can perform on the data."""
    plotting_df, comparison_df = load_data()
    
    print("=" * 70)
    print("BENCHMARK ANALYSIS EXAMPLES")
    print("=" * 70)
    
    # 1. Best performing methods overall
    print("\n1. TOP 5 METHODS BY RECALL@300")
    print("-" * 70)
    top5 = comparison_df.nlargest(5, "Recall@300_mean")
    cols_to_show = ["Method", "Recall@300_mean", "Time_mean", "AP_mean"]
    if "MRR@300_mean" in comparison_df.columns:
        cols_to_show.insert(3, "MRR@300_mean")
    print(top5[cols_to_show].to_string(index=False))
    
    # 2. Fastest methods
    print("\n2. TOP 5 FASTEST METHODS")
    print("-" * 70)
    fastest = comparison_df.nsmallest(5, "Time_mean")
    print(fastest[["Method", "Time_mean", "Recall@300_mean"]].to_string(index=False))
    
    # 3. Best speed/performance tradeoff (Recall@300 / Time)
    print("\n3. BEST SPEED/PERFORMANCE TRADEOFF")
    print("-" * 70)
    comparison_df["Efficiency"] = comparison_df["Recall@300_mean"] / comparison_df["Time_mean"]
    efficient = comparison_df.nlargest(5, "Efficiency")
    print(efficient[["Method", "Time_mean", "Recall@300_mean", "Efficiency"]].to_string(index=False))
    
    # 4. Network comparison for specific method
    print("\n4. NETWORK COMPARISON (QA Method)")
    print("-" * 70)
    qa_data = plotting_df[plotting_df["Method"] == "benchmark_qa"]
    if len(qa_data) > 0:
        network_comp = qa_data.groupby("Network").agg({
            "Time_s": "mean",
            "Genes_obtained_300": "mean",
            "AP": "mean",
        }).round(4)
        print(network_comp.to_string())
    else:
        print("No QA data found")
    
    # 5. Performance by number of seeds
    print("\n5. PERFORMANCE BY NUMBER OF SEEDS (Averaged across all methods)")
    print("-" * 70)
    by_seeds = plotting_df.groupby("Num_seeds").agg({
        "Time_s": "mean",
        "Genes_obtained_300": "mean",
        "AP": "mean",
    }).round(4)
    print(by_seeds.to_string())
    
    # 6. Correlation between metrics
    print("\n6. METRIC CORRELATIONS (Pearson)")
    print("-" * 70)
    metrics = ["Genes_obtained_300", "AP", "Accuracy", "Time_s"]
    corr_matrix = plotting_df[metrics].corr()
    print(corr_matrix.round(3).to_string())
    
    # 7. Methods with best MRR (if available)
    if "MRR@300_mean" in comparison_df.columns:
        print("\n7. TOP 5 METHODS BY MRR@300")
        print("-" * 70)
        top_mrr = comparison_df.nlargest(5, "MRR@300_mean")
        print(top_mrr[["Method", "MRR@300_mean", "Recall@300_mean", "AP_mean"]].to_string(index=False))
    else:
        print("\n7. MRR@300 data not available")
    
    # 8. QWalker methods analysis (if available)
    print("\n8. QWALKER METHODS ANALYSIS")
    print("-" * 70)
    qwalker_methods = comparison_df[comparison_df["Method"].str.contains("qwalker", case=False, na=False)]
    if len(qwalker_methods) > 0:
        cols_to_show = ["Method", "Time_mean", "Recall@300_mean", "AP_mean"]
        if "MRR@300_mean" in comparison_df.columns:
            cols_to_show.insert(3, "MRR@300_mean")
        print(qwalker_methods[cols_to_show].to_string(index=False))
    else:
        print("No QWalker methods found (may need to run benchmark with QWalker installed)")
    
    print("\n" + "=" * 70)


def export_custom_table():
    """Create a custom summary table for your specific needs."""
    plotting_df, _ = load_data()
    
    # Example: Create a pivot table for LaTeX/paper
    pivot = plotting_df.pivot_table(
        index="Method",
        columns="Network",
        values=["Genes_obtained_300", "Time_s"],
        aggfunc="mean",
    ).round(3)
    
    output_file = "benchmarking/custom_summary.csv"
    pivot.to_csv(output_file)
    print(f"\nCustom table saved to {output_file}")


def filter_by_criteria():
    """Filter methods by specific criteria."""
    plotting_df, _ = load_data()
    
    print("\n" + "=" * 70)
    print("METHODS MEETING SPECIFIC CRITERIA")
    print("=" * 70)
    
    # Find methods that are: fast (< 1s), accurate (Recall@300 > 0.5), good AP (> 0.2)
    filtered = plotting_df[
        (plotting_df["Time_s"] < 1.0) &
        (plotting_df["Genes_obtained_300"] > 0.5) &
        (plotting_df["AP"] > 0.2)
    ]
    
    if len(filtered) > 0:
        print("\nMethods: Time < 1s AND Recall@300 > 0.5 AND AP > 0.2")
        print("-" * 70)
        result = filtered.groupby("Method").agg({
            "Time_s": "mean",
            "Genes_obtained_300": "mean",
            "AP": "mean",
        }).round(4)
        print(result.to_string())
    else:
        print("\nNo methods meet all criteria. Try adjusting thresholds.")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    try:
        analyze_performance()
        export_custom_table()
        filter_by_criteria()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nMake sure to run 'python benchmarking/benchmark.py' first!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
