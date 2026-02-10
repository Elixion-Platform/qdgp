from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

OUTPUT_DIR = Path("benchmarking")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    res_df = pd.read_csv(OUTPUT_DIR / "benchmark.csv")
    mean_df = (
        res_df.groupby(["Method", "Num_seeds", "Network"], as_index=False)
        .agg({"Time (s)": ["mean", "std"], "Run": "count"})
    )
    mean_df.columns = [
        "Method",
        "Num_seeds",
        "Network",
        "Time_mean",
        "Time_std",
        "Runs",
    ]
    mean_df.to_csv(OUTPUT_DIR / "benchmark_summary.csv", index=False)
    # sns.lineplot(res_df, y="Time (s)", x="Num_seeds", hue="Method", row="Network")
    # grid = sns.FacetGrid(res_df, col="Network", col_wrap=2)
    # grid.map(sns.lineplot, kwargs={"x": "Num_seeds", "y": "Time (s)", "hue": "Method"})
    sns.relplot(
        res_df, y="Time (s)", x="Num_seeds", hue="Method", col="Network", kind="line"
    )
    # plt.show()
    plt.tight_layout()
    # plt.show()
    plt.savefig(OUTPUT_DIR / "benchmark.png")


if __name__ == "__main__":
    main()
