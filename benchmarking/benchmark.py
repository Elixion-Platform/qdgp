"""Unified benchmark system for disease gene prioritization methods.

This module provides a comprehensive benchmarking framework that evaluates both:
1. Computational Performance: Execution time for each method
2. Predictive Performance: Recall@k, MRR, AP, AUROC

METRICS ALIGNMENT WITH UPSTREAM (https://github.com/markgolds/qdgp.git):
- Recall@k: Identical to upstream (cumulative_hits / num_test_seeds)
- AP: Identical to upstream (sklearn.metrics.average_precision_score)
- AUROC: Identical to upstream (sklearn.metrics.roc_auc_score)
- MRR: Identical to upstream (methods ranked by True Hits per disease)
- True Hits@k: Added to support MRR computation (absolute gene count)

The benchmark:
- Splits disease seeds into train/test sets (50/50)
- Measures execution time for scoring
- Evaluates prediction quality on held-out test genes
- Computes MRR by ranking methods per disease (like upstream)
- Averages metrics across multiple runs and diseases
 (default: 10 repeats)

Output includes detailed per-disease results and aggregated summaries.
See UNIFIED_BENCHMARK.md and UPSTREAM_ALIGNMENT.md for full documentation.
"""

import logging
import sys
import argparse
from functools import wraps
from pathlib import Path
from time import time
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

import qdgp.data as dt
import qdgp.models as md
import qdgp.utils as ut


def _try_import_qwalker() -> Tuple[bool, Optional[Callable], Optional[Callable]]:
    """Attempt to import QWalker from the sibling repo in this workspace."""
    try:
        from qwalker.walks import quantum_walks as qwalker_quantum
        from qwalker.walks import random_walks as qwalker_random

        return True, qwalker_random, qwalker_quantum
    except Exception:
        # Try the typical workspace layout: ../QWalker/src
        repo_root = Path(__file__).resolve().parents[1]  # .../qdgp
        qwalker_src = repo_root.parent / "QWalker" / "src"
        if qwalker_src.exists():
            sys.path.insert(0, str(qwalker_src))
            try:
                from qwalker.walks import quantum_walks as qwalker_quantum
                from qwalker.walks import random_walks as qwalker_random

                return True, qwalker_random, qwalker_quantum
            except Exception:
                return False, None, None
        return False, None, None

logger = logging.getLogger(__name__)
formatter = logging.Formatter("%(message)s")
OUTPUT_DIR = Path("benchmarking")
COLLECTRI_DIR = Path("data") / "collectri"

# Global shuffle for consistent tie-breaking
RNG = np.random.default_rng(0)

def timing_and_evaluation(f: Callable) -> Callable:
    """Decorator that measures time and returns evaluation metrics."""
    @wraps(f)
    def wrap(*args: Dict, **kw: Dict) -> Dict[str, float]:
        ts = time()
        results = f(*args, **kw)
        te = time()
        results["time"] = te - ts
        return results

    return wrap


def evaluate_scores(
    scores: np.ndarray,
    train_seeds: List[int],
    test_seeds: List[int],
    n_nodes: int,
    shuffled_nodes: np.ndarray,
    top_k: List[int] = [25, 100, 300],
) -> Dict[str, float]:
    """Evaluate scores and compute Recall@k, True Hits@k, AP, AUROC.
    
    Metrics align with upstream qdgp repository:
    - Recall@k: Fraction of test genes recovered in top k predictions
    - True Hits@k: Absolute number of test genes found in top k
    - AP: Average precision (sklearn implementation)
    - AUROC: Area under ROC curve
    
    Note: MRR is computed separately by ranking methods per disease (see post-processing).
    """
    # Convert scores to a 1D numpy array of length n_nodes.
    # - Native qdgp models typically return ndarray-like.
    # - QWalker methods return dict[node] -> probability.
    # - Some implementations may wrap the actual array in a length-1 container.
    if isinstance(scores, (list, tuple)) and len(scores) == 1:
        scores = scores[0]

    if isinstance(scores, dict):
        dense = np.zeros(int(n_nodes), dtype=float)
        bad = 0
        for node, val in scores.items():
            try:
                idx = int(node)
                if 0 <= idx < int(n_nodes):
                    dense[idx] = float(val)
                else:
                    bad += 1
            except Exception:
                bad += 1
        # If most keys were unusable, raise a helpful error.
        if bad > 0 and (len(scores) > 0) and (bad / float(len(scores)) > 0.5):
            raise ValueError(
                "Scores dict keys are not compatible with node indexing; "
                f"bad_keys={bad}/{len(scores)}"
            )
        scores = dense

    if hasattr(scores, "toarray"):  # scipy sparse matrix
        scores = scores.toarray()

    scores = np.asarray(scores)
    # Unwrap object array containing a single array-like payload.
    if scores.dtype == object and scores.size == 1:
        payload = scores.item()
        if hasattr(payload, "toarray"):
            payload = payload.toarray()
        scores = np.asarray(payload)

    scores = scores.ravel()
    if scores.shape[0] != int(n_nodes):
        raise ValueError(
            f"Expected {n_nodes} scores but got {scores.shape[0]} (type={type(scores).__name__})"
        )
    
    train_seed_mask = ut.seed_list_to_mask(train_seeds, n_nodes)
    test_mask = (1 - train_seed_mask).astype(bool)
    
    scores_filtered = scores[test_mask]
    y_true = ut.seed_list_to_mask(test_seeds, n_nodes)[test_mask]
    
    # Order labels by scores, breaking ties with shuffle (same as upstream)
    ordered_labels = [
        y for _, _, y in sorted(
            zip(scores_filtered, shuffled_nodes[test_mask], y_true),
            reverse=True,
        )
    ]
    
    cumulative_hits = np.cumsum(ordered_labels)
    num_test_seeds = len(test_seeds)
    
    # Compute Recall@k and True Hits@k (aligned with upstream)
    metrics = {}
    for k in top_k:
        if k <= len(cumulative_hits):
            true_hits_at_k = float(cumulative_hits[k - 1])
            recall_at_k = true_hits_at_k / num_test_seeds
        else:
            true_hits_at_k = float(cumulative_hits[-1])
            recall_at_k = true_hits_at_k / num_test_seeds
        
        metrics[f"True_Hits@{k}"] = true_hits_at_k
        metrics[f"Recall@{k}"] = recall_at_k
    
    # Compute AP and AUROC (same as upstream)
    metrics["AP"] = float(average_precision_score(y_true, scores_filtered))
    metrics["AUROC"] = float(roc_auc_score(y_true, scores_filtered))
    
    return metrics


def _summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate benchmark results across runs."""
    metric_cols = [c for c in df.columns if c not in ["Method", "Disease", "Num_seeds", "Num_train_seeds", "Run", "Network", "Timestamp"]]
    
    agg_dict = {col: ["mean", "std"] for col in metric_cols}
    
    summary = (
        df.groupby(["Method", "Num_seeds", "Network"], as_index=False)
        .agg(agg_dict)
    )
    
    # Add count of runs
    run_counts = df.groupby(["Method", "Num_seeds", "Network"]).size().reset_index(name="Runs")
    
    # Flatten column names
    new_cols = ["Method", "Num_seeds", "Network"]
    for col in metric_cols:
        new_cols.extend([f"{col}_mean", f"{col}_std"])
    summary.columns = new_cols
    
    # Merge with run counts
    summary = summary.merge(run_counts, on=["Method", "Num_seeds", "Network"])
    
    return summary


def _create_plotting_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create a simplified table optimized for plotting with key metrics.
    
    Columns: Method, Network, Num_seeds, Time_s, True_Hits@25/100/300,
             Genes_obtained_25/100/300 (Recall), Accuracy (AUROC), AP
    """
    plot_df = df.groupby(["Method", "Network", "Num_seeds"], as_index=False).agg({
        "Time (s)": "mean",
        "True_Hits@25": "mean",
        "True_Hits@100": "mean",
        "True_Hits@300": "mean",
        "Recall@25": "mean",
        "Recall@100": "mean",
        "Recall@300": "mean",
        "AP": "mean",
        "AUROC": "mean",
        "Num_train_seeds": "mean",
    })
    
    # Rename columns for clarity
    plot_df = plot_df.rename(columns={
        "Time (s)": "Time_s",
        "True_Hits@25": "True_Hits_25",
        "True_Hits@100": "True_Hits_100",
        "True_Hits@300": "True_Hits_300",
        "Recall@25": "Genes_obtained_25",
        "Recall@100": "Genes_obtained_100",
        "Recall@300": "Genes_obtained_300",
        "AUROC": "Accuracy",
        "Num_train_seeds": "Num_train_seeds",
    })
    
    # Reorder columns for better readability
    column_order = [
        "Method", "Network", "Num_seeds", "Num_train_seeds",
        "Time_s", 
        "True_Hits_25", "True_Hits_100", "True_Hits_300",
        "Genes_obtained_25", "Genes_obtained_100", "Genes_obtained_300",
        "Accuracy", "AP"
    ]
    return plot_df[column_order]


def _compute_mrr(df: pd.DataFrame, iteration: int = 300) -> pd.DataFrame:
    """Compute MRR aligned with upstream: rank methods by True Hits per disease.
    
    This matches the upstream qdgp repository's MRR computation:
    For each disease, rank methods by their True Hits at the given iteration,
    then compute MRR = 1/rank, and average across diseases.
    """
    col_name = f"True_Hits@{iteration}"
    if col_name not in df.columns:
        logger.warning("Column %s not found, cannot compute MRR@%d", col_name, iteration)
        return pd.DataFrame()
    
    # Average across runs first (if multiple runs per method/disease)
    tdf = df.groupby(["Method", "Disease"], as_index=False).agg({col_name: "mean"})
    
    # Rank methods by True Hits for each disease
    tdf["Rank"] = tdf.groupby("Disease")[col_name].rank(ascending=False, method="min")
    tdf["MRR"] = 1.0 / tdf["Rank"]
    
    # Aggregate by Method
    mrr_df = tdf.groupby("Method", as_index=False).agg({"MRR": ["mean", "std"]})
    mrr_df.columns = ["Method", f"MRR@{iteration}_mean", f"MRR@{iteration}_std"]
    
    return mrr_df


def _create_method_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Create a method comparison table averaged across all networks and diseases."""
    comparison = df.groupby(["Method"], as_index=False).agg({
        "Time (s)": ["mean", "std"],
        "True_Hits@25": ["mean", "std"],
        "True_Hits@100": ["mean", "std"],
        "True_Hits@300": ["mean", "std"],
        "Recall@25": ["mean", "std"],
        "Recall@100": ["mean", "std"],
        "Recall@300": ["mean", "std"],
        "AP": ["mean", "std"],
        "AUROC": ["mean", "std"],
        "Num_seeds": "mean",
    })
    
    # Flatten column names
    comparison.columns = [
        "Method",
        "Time_mean", "Time_std",
        "True_Hits@25_mean", "True_Hits@25_std",
        "True_Hits@100_mean", "True_Hits@100_std",
        "True_Hits@300_mean", "True_Hits@300_std",
        "Recall@25_mean", "Recall@25_std",
        "Recall@100_mean", "Recall@100_std",
        "Recall@300_mean", "Recall@300_std",
        "AP_mean", "AP_std",
        "AUROC_mean", "AUROC_std",
        "Avg_Num_seeds",
    ]
    
    return comparison.sort_values("Recall@300_mean", ascending=False)


def _synthetic_seeds_by_group(
    nodes: Iterable[int],
    seed_sizes: List[int],
    diseases_per_size: int = 10,
    rng_seed: int = 0,
) -> Dict[str, List[int]]:
    nodes = list(nodes)
    import numpy as np

    rng = np.random.default_rng(int(rng_seed))
    seeds_by = {}
    for k in seed_sizes:
        k = int(k)
        if k <= 0:
            continue
        if k > len(nodes):
            continue
        for i in range(int(diseases_per_size)):
            seeds = rng.choice(nodes, size=k, replace=False).tolist()
            seeds_by[f"synthetic_k{k}_{i:02d}"] = seeds
    return seeds_by


def _read_deg_table(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    cols = {c.lower(): c for c in df.columns}
    required = {"gene", "log2foldchange", "padj"}
    if not required.issubset(cols.keys()):
        missing = required.difference(cols.keys())
        raise ValueError(f"Missing DEG columns in {path.name}: {sorted(missing)}")
    df = df[[cols["gene"], cols["log2foldchange"], cols["padj"]]].copy()
    df.columns = ["gene", "log2FoldChange", "padj"]
    return df


def _collectri_deg_seeds(
    code_dict: Dict[str, int],
    max_genes: Optional[int] = None,
    padj_threshold: float = 0.05,
) -> Tuple[Dict[str, List[int]], Dict[str, List[str]]]:
    deg_files = {
        "deg_er": COLLECTRI_DIR / "DEG" / "Malignant_ER_vs_Normal.xlsx",
        "deg_her2": COLLECTRI_DIR / "DEG" / "Malignant_HER2_vs_Normal.xlsx",
        "deg_tnbc": COLLECTRI_DIR / "DEG" / "Malignant_TNBC_vs_Normal.xlsx",
    }

    seeds_by_disease: Dict[str, List[int]] = {}
    genes_by_disease: Dict[str, List[str]] = {}
    combined_scores: Dict[str, float] = {}

    for label, path in deg_files.items():
        df = _read_deg_table(path)
        df = df[df["padj"].notna() & df["log2FoldChange"].notna()]
        df = df[df["padj"] < float(padj_threshold)]
        df = df.sort_values("log2FoldChange", ascending=False)
        genes = df["gene"].astype(str).tolist()
        if max_genes is not None:
            genes = genes[: int(max_genes)]

        mapped = [code_dict[g] for g in genes if g in code_dict]
        if len(mapped) == 0:
            continue

        genes_by_disease[label] = genes
        seeds_by_disease[label] = mapped

        for gene, log2fc in zip(df["gene"].astype(str), df["log2FoldChange"]):
            if pd.isna(log2fc) or gene not in code_dict:
                continue
            score = float(log2fc)
            if gene not in combined_scores or score > combined_scores[gene]:
                combined_scores[gene] = score

    if combined_scores:
        combined = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        if max_genes is not None:
            combined = combined[: int(max_genes)]
        combined_genes = [gene for gene, _ in combined]
        combined_ids = [code_dict[g] for g in combined_genes if g in code_dict]
        if len(combined_ids) > 0:
            genes_by_disease["deg_combined"] = combined_genes
            seeds_by_disease["deg_combined"] = combined_ids

    return seeds_by_disease, genes_by_disease


def _collectri_validation_genes(code_dict: Dict[str, int]) -> pd.DataFrame:
    path = COLLECTRI_DIR / "high_confidence_genes" / "BC High-confidence genes.xlsx"
    df = pd.read_excel(path, sheet_name=0)
    cols = {c.lower(): c for c in df.columns}
    gene_col = cols.get("genes") or cols.get("gene")
    if gene_col is None:
        raise ValueError("Missing 'Genes' column in high-confidence genes file")
    genes = df[gene_col].astype(str).dropna().tolist()
    valid_genes = [g for g in genes if g in code_dict]
    mapped = [code_dict[g] for g in valid_genes]
    return pd.DataFrame({"gene": valid_genes, "node_id": mapped})


@timing_and_evaluation
def benchmark_qa(G, nl, diseases, seeds_by_disease, train_seeds_by_disease, test_seeds_by_disease, shuffled_nodes) -> Dict[str, float]:
    A = nx.adjacency_matrix(G, nodelist=nl)
    all_metrics = {}
    for disease in diseases:
        train_seeds = train_seeds_by_disease[disease]
        test_seeds = test_seeds_by_disease[disease]
        scores = md.qa_score(G, train_seeds, t=0.45, H=A, diag=None)
        metrics = evaluate_scores(scores, train_seeds, test_seeds, G.number_of_nodes(), shuffled_nodes)
        for k, v in metrics.items():
            all_metrics.setdefault(k, []).append(v)
    # Return average for most metrics, but keep True Hits for MRR computation
    result = {}
    for k, values in all_metrics.items():
        result[k] = float(np.mean(values))
    return result


@timing_and_evaluation
def benchmark_crw(G, nl, diseases, seeds_by_disease, train_seeds_by_disease, test_seeds_by_disease, shuffled_nodes) -> Dict[str, float]:
    L = nx.laplacian_matrix(G, nodelist=nl)
    all_metrics = {}
    for disease in diseases:
        train_seeds = train_seeds_by_disease[disease]
        test_seeds = test_seeds_by_disease[disease]
        scores = md.dk_score(G, train_seeds, L=L, t=0.3)
        metrics = evaluate_scores(scores, train_seeds, test_seeds, G.number_of_nodes(), shuffled_nodes)
        for k, v in metrics.items():
            all_metrics.setdefault(k, []).append(v)
    result = {}
    for k, values in all_metrics.items():
        result[k] = float(np.mean(values))
    return result


@timing_and_evaluation
def benchmark_rwr(G, nl, diseases, seeds_by_disease, train_seeds_by_disease, test_seeds_by_disease, shuffled_nodes) -> Dict[str, float]:
    A = nx.adjacency_matrix(G, nodelist=nl)
    R = md.normalize_adjacency(G, A)  # for random walk with restart
    all_metrics = {}
    for disease in diseases:
        train_seeds = train_seeds_by_disease[disease]
        test_seeds = test_seeds_by_disease[disease]
        scores = md.rwr_score(G, train_seeds, normalized_adjacency=R, return_prob=0.4)
        metrics = evaluate_scores(scores, train_seeds, test_seeds, G.number_of_nodes(), shuffled_nodes)
        for k, v in metrics.items():
            all_metrics.setdefault(k, []).append(v)
    result = {}
    for k, values in all_metrics.items():
        result[k] = float(np.mean(values))
    return result


@timing_and_evaluation
def benchmark_dia(G, nl, diseases, seeds_by_disease, train_seeds_by_disease, test_seeds_by_disease, shuffled_nodes) -> Dict[str, float]:
    A = nx.adjacency_matrix(G, nodelist=nl)
    A_d = A.toarray()
    all_metrics = {}
    for disease in diseases:
        train_seeds = train_seeds_by_disease[disease]
        test_seeds = test_seeds_by_disease[disease]
        scores = md.diamond_score(G, train_seeds, A=A_d, alpha=9, number_to_rank=100)
        metrics = evaluate_scores(scores, train_seeds, test_seeds, G.number_of_nodes(), shuffled_nodes)
        for k, v in metrics.items():
            all_metrics.setdefault(k, []).append(v)
    result = {}
    for k, values in all_metrics.items():
        result[k] = float(np.mean(values))
    return result


@timing_and_evaluation
def benchmark_nei(G, nl, diseases, seeds_by_disease, train_seeds_by_disease, test_seeds_by_disease, shuffled_nodes) -> Dict[str, float]:
    A = nx.adjacency_matrix(G, nodelist=nl)
    A_d = A.toarray()
    all_metrics = {}
    for disease in diseases:
        train_seeds = train_seeds_by_disease[disease]
        test_seeds = test_seeds_by_disease[disease]
        scores = md.neighbourhood_score(G, train_seeds, A=A_d)
        metrics = evaluate_scores(scores, train_seeds, test_seeds, G.number_of_nodes(), shuffled_nodes)
        for k, v in metrics.items():
            all_metrics.setdefault(k, []).append(v)
    result = {}
    for k, values in all_metrics.items():
        result[k] = float(np.mean(values))
    return result


def _make_benchmark_qwalker_random(
    qwalker_random: Callable,
    *,
    mode: str,
    restart_prob: float,
    n_steps: int,
    n_walkers: Optional[int] = None,
    rng_seed: Optional[int] = None,
) -> Callable:
    @timing_and_evaluation
    def _bench(G, nl, diseases, seeds_by_disease, train_seeds_by_disease, test_seeds_by_disease, shuffled_nodes) -> Dict[str, float]:
        all_metrics = {}
        for disease in diseases:
            train_seeds = train_seeds_by_disease[disease]
            test_seeds = test_seeds_by_disease[disease]
            scores = qwalker_random(
                G,
                seed_list=train_seeds,
                restart_prob=float(restart_prob),
                n_steps=int(n_steps),
                mode=str(mode),
                n_walkers=int(n_walkers) if n_walkers is not None else 1000,
                rng_seed=rng_seed,
            )
            metrics = evaluate_scores(scores, train_seeds, test_seeds, G.number_of_nodes(), shuffled_nodes)
            for k, v in metrics.items():
                all_metrics.setdefault(k, []).append(v)
        result = {}
        for k, values in all_metrics.items():
            result[k] = float(np.mean(values))
        return result

    # Make the method names stable for grouping in plots
    _bench.__name__ = (
        f"qwalker_rw_{mode}_rp{restart_prob:g}_steps{int(n_steps)}"
        + (f"_n{int(n_walkers)}" if n_walkers is not None else "")
    )
    return _bench


def _make_benchmark_qwalker_quantum(
    qwalker_quantum: Callable,
    *,
    t: float,
    hamiltonian: str,
) -> Callable:
    @timing_and_evaluation
    def _bench(G, nl, diseases, seeds_by_disease, train_seeds_by_disease, test_seeds_by_disease, shuffled_nodes) -> Dict[str, float]:
        all_metrics = {}
        for disease in diseases:
            train_seeds = train_seeds_by_disease[disease]
            test_seeds = test_seeds_by_disease[disease]
            scores = qwalker_quantum(
                G,
                times=float(t),
                hamiltonian=str(hamiltonian),
                seed_list=train_seeds,
            )
            metrics = evaluate_scores(scores, train_seeds, test_seeds, G.number_of_nodes(), shuffled_nodes)
            for k, v in metrics.items():
                all_metrics.setdefault(k, []).append(v)
        result = {}
        for k, values in all_metrics.items():
            result[k] = float(np.mean(values))
        return result

    _bench.__name__ = f"qwalker_qw_{hamiltonian}_t{float(t):g}"
    return _bench


def main(network: str, *, n_runs: int = 10, split_ratio: float = 0.5) -> pd.DataFrame:
    if network == "collectri":
        G, code_dict = dt.build_graph_collectri(
            Path("data"), filter_method=dt.FilterGCC.TRUE
        )
        seeds_by_disease, _ = _collectri_deg_seeds(code_dict)
        diseases = list(seeds_by_disease.keys())
        if len(diseases) == 0:
            seeds_by_disease = _synthetic_seeds_by_group(
                G.nodes(), seed_sizes=[15, 30, 60, 120], diseases_per_size=10, rng_seed=0
            )
            diseases = list(seeds_by_disease.keys())
        else:
            try:
                validation_df = _collectri_validation_genes(code_dict)
                validation_df.to_csv(
                    OUTPUT_DIR / "collectri_validation_genes.csv", index=False
                )
            except Exception as exc:
                logger.warning("Collectri validation genes unavailable: %s", exc)
    else:
        G, code_dict, seeds_by_disease = dt.load_dataset(
            "gmb", network, dt.FilterGCC.TRUE
        )
        diseases = list(seeds_by_disease.keys())
        if len(diseases) == 0:
            # Networks like CollecTRI won't have matching disease annotations.
            seeds_by_disease = _synthetic_seeds_by_group(
                G.nodes(), seed_sizes=[15, 30, 60, 120], diseases_per_size=10, rng_seed=0
            )
            diseases = list(seeds_by_disease.keys())

    n = G.number_of_nodes()
    nl = range(n)
    n_runs = int(n_runs)
    split_ratio = float(split_ratio)
    shuffled_nodes = RNG.permutation(list(G.nodes()))
    
    rows = []
    funcs: List[Callable] = [
        benchmark_nei,
        benchmark_dia,
        benchmark_crw,
        benchmark_rwr,
        benchmark_qa,
    ]

    has_qwalker, qwalker_random, qwalker_quantum = _try_import_qwalker()
    if has_qwalker and qwalker_random is not None and qwalker_quantum is not None:
        # Classical (QWalker) parameter sweeps
        funcs.extend(
            [
                _make_benchmark_qwalker_random(
                    qwalker_random,
                    mode="matrix",
                    restart_prob=0.15,
                    n_steps=50,
                ),
                _make_benchmark_qwalker_random(
                    qwalker_random,
                    mode="matrix",
                    restart_prob=0.40,
                    n_steps=100,
                ),
                _make_benchmark_qwalker_random(
                    qwalker_random,
                    mode="mc",
                    restart_prob=0.15,
                    n_steps=50,
                    n_walkers=2000,
                    rng_seed=0,
                ),
            ]
        )

        # Quantum (QWalker) parameter sweeps
        funcs.extend(
            [
                _make_benchmark_qwalker_quantum(
                    qwalker_quantum, t=0.5, hamiltonian="adjacency"
                ),
                _make_benchmark_qwalker_quantum(
                    qwalker_quantum, t=1.0, hamiltonian="adjacency"
                ),
                _make_benchmark_qwalker_quantum(
                    qwalker_quantum, t=1.0, hamiltonian="laplacian"
                ),
            ]
        )
    else:
        logger.info("QWalker not importable; skipping QWalker benchmark methods.")
    for run in range(n_runs):
        # Create train/test splits for this run
        train_seeds_by_disease = {}
        test_seeds_by_disease = {}
        for disease, seeds in seeds_by_disease.items():
            if len(seeds) < 4:  # Need at least 4 seeds for reasonable split
                continue
            train_seeds, test_seeds = train_test_split(
                seeds, train_size=split_ratio, random_state=run
            )
            train_seeds_by_disease[disease] = train_seeds
            test_seeds_by_disease[disease] = test_seeds
        
        filtered_diseases = list(train_seeds_by_disease.keys())
        
        for i, dis in enumerate(filtered_diseases):
            for f in funcs:
                res = f(G, nl, [dis], seeds_by_disease, train_seeds_by_disease, test_seeds_by_disease, shuffled_nodes)
                row = [
                    f.__name__, 
                    dis, 
                    len(seeds_by_disease[dis]),
                    len(train_seeds_by_disease[dis]),
                    res.get("time", 0),
                    res.get("True_Hits@25", 0),
                    res.get("True_Hits@100", 0),
                    res.get("True_Hits@300", 0),
                    res.get("Recall@25", 0),
                    res.get("Recall@100", 0),
                    res.get("Recall@300", 0),
                    res.get("AP", 0),
                    res.get("AUROC", 0),
                    run,
                    network,
                ]
                rows.append(row)
                logger.info(
                    "%s - %s - %d/%d | seeds=%d train=%d | time=%.3f s | Recall: @25=%.3f @100=%.3f @300=%.3f | AP=%.3f AUROC=%.3f | run=%d net=%s",
                    f.__name__,
                    dis[:20],
                    i + 1,
                    len(filtered_diseases),
                    len(seeds_by_disease[dis]),
                    len(train_seeds_by_disease[dis]),
                    res.get("time", 0),
                    res.get("Recall@25", 0),
                    res.get("Recall@100", 0),
                    res.get("Recall@300", 0),
                    res.get("AP", 0),
                    res.get("AUROC", 0),
                    run,
                    network,
                )

    df = pd.DataFrame(
        rows, 
        columns=[
            "Method", 
            "Disease", 
            "Num_seeds",
            "Num_train_seeds",
            "Time (s)", 
            "True_Hits@25",
            "True_Hits@100",
            "True_Hits@300",
            "Recall@25",
            "Recall@100",
            "Recall@300",
            "AP",
            "AUROC",
            "Run", 
            "Network",
        ]
    )
    # Add metadata for easier analysis
    df["Timestamp"] = pd.Timestamp.now()
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified benchmark (timing + performance metrics)."
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help=(
            "Number of repeated 50/50 train-test splits per disease "
            "(default: 10, as in the original publication)."
        ),
    )
    args = parser.parse_args()

    if not Path("logs").exists():
        Path("logs").mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename="logs/benchmark.log",
        filemode="w",
        level=logging.INFO,
    )

    # Create a handler
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)
    logger.info("Starting unified benchmark (timing + performance metrics).")
    all_dfs = []
    for network in ["wl", "collectri"]:
        net_df = main(network, n_runs=int(args.runs))
        all_dfs.append(net_df)
    DF = pd.concat(all_dfs)
    
    # Export multiple CSV formats for different uses
    output_csv = OUTPUT_DIR / "benchmark.csv"
    summary_csv = OUTPUT_DIR / "benchmark_summary.csv"
    plotting_csv = OUTPUT_DIR / "benchmark_plotting.csv"
    comparison_csv = OUTPUT_DIR / "benchmark_method_comparison.csv"
    mrr25_csv = OUTPUT_DIR / "benchmark_mrr25.csv"
    mrr300_csv = OUTPUT_DIR / "benchmark_mrr300.csv"
    
    # 1. Complete raw data (all runs, all diseases)
    DF.to_csv(output_csv, index=False)
    logger.info("✓ Saved complete results to %s", output_csv)
    
    # 2. Summary statistics (mean/std by method, num_seeds, network)
    summary_df = _summarize_results(DF)
    summary_df.to_csv(summary_csv, index=False)
    logger.info("✓ Saved aggregated summary to %s", summary_csv)
    
    # 3. Plotting-optimized table (averaged, key metrics only)
    plotting_df = _create_plotting_table(DF)
    plotting_df.to_csv(plotting_csv, index=False)
    logger.info("✓ Saved plotting table to %s", plotting_csv)
    
    # 4. Method comparison (overall averages across all conditions)
    comparison_df = _create_method_comparison(DF)
    comparison_df.to_csv(comparison_csv, index=False)
    logger.info("✓ Saved method comparison to %s", comparison_csv)
    
    # 5. MRR tables (aligned with upstream - rank methods by True Hits per disease)
    mrr25_df = _compute_mrr(DF, iteration=25)
    if not mrr25_df.empty:
        mrr25_df.to_csv(mrr25_csv, index=False)
        logger.info("✓ Saved MRR@25 to %s", mrr25_csv)
    
    mrr300_df = _compute_mrr(DF, iteration=300)
    if not mrr300_df.empty:
        mrr300_df.to_csv(mrr300_csv, index=False)
        logger.info("✓ Saved MRR@300 to %s", mrr300_csv)
        
        # Merge MRR into comparison table
        comparison_df = comparison_df.merge(mrr300_df, on="Method", how="left")
        comparison_df.to_csv(comparison_csv, index=False)
    
    # Print summary of key metrics
    logger.info("\n" + "="*60)
    logger.info("BENCHMARK SUMMARY - Top 5 Methods by Recall@300")
    logger.info("="*60)
    display_cols = ["Method", "Time_mean", "Recall@300_mean", "AP_mean"]
    if "MRR@300_mean" in comparison_df.columns:
        display_cols.insert(3, "MRR@300_mean")
    top_methods = comparison_df.head(5)[display_cols]
    logger.info("\n" + top_methods.to_string(index=False))
    logger.info("\n" + "="*60)
