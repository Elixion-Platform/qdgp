import logging
import sys
from functools import wraps
from pathlib import Path
from time import time
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import networkx as nx
import pandas as pd

import qdgp.data as dt
import qdgp.models as md


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


def timing(f: Callable) -> Callable:
    @wraps(f)
    def wrap(*args: Dict, **kw: Dict) -> float:
        ts = time()
        _ = f(*args, **kw)
        te = time()
        f_name = f.__name__
        diff = te - ts
        return diff

    return wrap


def _summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["Method", "Num_seeds", "Network"], as_index=False)
        .agg({"Time (s)": ["mean", "std"], "Run": "count"})
        .rename(columns={"count": "Runs"})
    )
    summary.columns = [
        "Method",
        "Num_seeds",
        "Network",
        "Time_mean",
        "Time_std",
        "Runs",
    ]
    return summary


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


@timing
def benchmark_qa(G, nl, diseases, seeds_by_disease) -> None:
    A = nx.adjacency_matrix(G, nodelist=nl)
    for disease in diseases:
        seeds = seeds_by_disease[disease]
        md.qa_score(G, seeds, t=0.45, H=A, diag=None)


@timing
def benchmark_crw(G, nl, diseases, seeds_by_disease) -> None:
    L = nx.laplacian_matrix(G, nodelist=nl)
    for disease in diseases:
        seeds = seeds_by_disease[disease]
        md.dk_score(G, seeds, L=L, t=0.3)


@timing
def benchmark_rwr(G, nl, diseases, seeds_by_disease) -> None:
    A = nx.adjacency_matrix(G, nodelist=nl)
    # A_d = A.toarray()
    # R = md.normalize_adjacency(G, A_d)
    R = md.normalize_adjacency(G, A)  # for random walk with restart
    for disease in diseases:
        seeds = seeds_by_disease[disease]
        md.rwr_score(G, seeds, normalized_adjacency=R, return_prob=0.4)


@timing
def benchmark_dia(G, nl, diseases, seeds_by_disease) -> None:
    A = nx.adjacency_matrix(G, nodelist=nl)
    A_d = A.toarray()
    for disease in diseases:
        seeds = seeds_by_disease[disease]
        md.diamond_score(G, seeds, A=A_d, alpha=9, number_to_rank=100)


@timing
def benchmark_nei(G, nl, diseases, seeds_by_disease) -> None:
    A = nx.adjacency_matrix(G, nodelist=nl)
    A_d = A.toarray()
    for disease in diseases:
        seeds = seeds_by_disease[disease]
        md.neighbourhood_score(G, seeds, A=A_d)


def _make_benchmark_qwalker_random(
    qwalker_random: Callable,
    *,
    mode: str,
    restart_prob: float,
    n_steps: int,
    n_walkers: Optional[int] = None,
    rng_seed: Optional[int] = None,
) -> Callable:
    @timing
    def _bench(G, nl, diseases, seeds_by_disease) -> None:
        for disease in diseases:
            seeds = seeds_by_disease[disease]
            qwalker_random(
                G,
                seed_list=seeds,
                restart_prob=float(restart_prob),
                n_steps=int(n_steps),
                mode=str(mode),
                n_walkers=int(n_walkers) if n_walkers is not None else 1000,
                rng_seed=rng_seed,
            )

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
    @timing
    def _bench(G, nl, diseases, seeds_by_disease) -> None:
        for disease in diseases:
            seeds = seeds_by_disease[disease]
            qwalker_quantum(
                G,
                times=float(t),
                hamiltonian=str(hamiltonian),
                seed_list=seeds,
            )

    _bench.__name__ = f"qwalker_qw_{hamiltonian}_t{float(t):g}"
    return _bench


def main(network: str) -> pd.DataFrame:
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
    n_runs = 5
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
        for i, dis in enumerate(diseases[:]):
            for f in funcs:
                res = f(G, nl, [dis], seeds_by_disease)
                rows.append(
                    [f.__name__, dis, len(seeds_by_disease[dis]), res, run, network]
                )
                logger.info(
                    "%s - %s - %d - %d %f %d %s",
                    f.__name__,
                    dis[:4],
                    i,
                    len(seeds_by_disease[dis]),
                    res,
                    run,
                    network,
                )

    return pd.DataFrame(
        rows, columns=["Method", "Disease", "Num_seeds", "Time (s)", "Run", "Network"]
    )


if __name__ == "__main__":
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
    logger.info("Starting benchmark.")
    all_dfs = []
    for network in ["wl", "collectri"]:
        net_df = main(network)
        all_dfs.append(net_df)
    DF = pd.concat(all_dfs)
    output_csv = OUTPUT_DIR / "benchmark.csv"
    summary_csv = OUTPUT_DIR / "benchmark_summary.csv"
    DF.to_csv(output_csv, index=False)
    _summarize_results(DF).to_csv(summary_csv, index=False)
    logger.info("Saved benchmark results to %s", output_csv)
    logger.info("Saved benchmark summary to %s", summary_csv)
