# QDGP - Quantum Disease Gene Prioritisation

## Overview

**QDGP** is an advanced computational framework for disease gene prioritization using quantum walk algorithms. It implements methods to identify disease-associated genes by simulating quantum and classical walks on protein-protein interaction (PPI) networks seeded with known disease genes.

This repository is an enhanced version of the original [qdgp repository](https://github.com/markgolds/qdgp) by Mark Goldsmith, which implements the research from the paper [Disease Gene Prioritization With Quantum Walks](https://arxiv.org/abs/2311.05486).

### What This Repository Contains

- **Core Algorithms**: Quantum walk (QA), classical random walk (DK), and neighborhood-based scoring methods for gene prioritization
- **Multiple PPI Networks**: Support for 8 different protein-protein interaction networks:
  - GMB, BioGRID, STRING, IID, APID, HPRD, Whitmore-Leblond (WL), and CollecTRI
- **Disease Gene Sets**: Three disease datasets:
  - GMB (Gene-disease associations from graph mining)
  - Open Targets (OT)
  - DisGeNET (DGN)
- **Comprehensive Benchmarking Framework**: Tools to evaluate and compare multiple gene prioritization methods
- **Cross-Validation Pipeline**: Systematic evaluation across different network/disease combinations

### What It Does

The framework scores genes for a given disease by:
1. Taking seed genes (known disease-associated genes) as input
2. Running quantum or classical walks on a PPI network starting from these seeds
3. Computing scores for all genes based on walk probability/amplitude
4. Ranking genes by score to generate predictions of likely disease-associated genes

### Output

The framework produces:
- **Ranked gene predictions**: CSV files with top N predicted genes and their scores
- **Cross-validation results**: Performance metrics (Recall@k, MRR, AP, AUROC) across multiple runs
- **Benchmarking reports**: Comparative analysis of different algorithms with both computational and predictive performance
- **Visualizations**: Plots comparing method performance across networks and diseases

### Key Enhancements Over Original Repository

This version includes several important improvements:

1. **QWalker Integration**: Integration with the quantum walk framework from the [QWalker repository](https://github.com/markgolds/qwalker) for enhanced quantum algorithms
2. **Unified Benchmarking System**: Complete benchmark suite that captures both:
   - Computational performance (execution time)
   - Predictive performance (Recall@k, MRR, AP, AUROC)
3. **Enhanced Metrics**: Aligned metrics with upstream repository including True Hits@k for better method comparison
4. **Improved Visualization**: Better analysis and plotting tools for benchmark results
5. **CollecTRI Support**: Added support for the CollecTRI gene regulatory network
6. **Extended Documentation**: Detailed benchmarking guides and metrics documentation

## Installation
0. Install [Miniconda](https://docs.anaconda.com/free/miniconda/) if it's not already installed.

1. Clone this repository:
   ```
   git clone https://github.com/MarkEGold/qdgp
   ```
2. Create a Conda environment from the `qdgp.yaml` file in the root of the repository:
   ```
   conda env create -f qdgp.yaml
   ```
   Note that this environment is called `qdgp`. The new environment will
   contain Poetry.

3. Activate the new environment:
   ```
   conda activate qdgp
   ```
4. Use poetry to install required packages:
   ```
   python -m poetry install
   ```

## Usage

The two main programs of interest are 

- `cross-validate.py`, for doing cross-validation accross several models, as in the paper, and
	
- `predict.py`, for making predictions using the quantum walk method described in the corresponding paper.

### Cross-validation

Statistically similar results to those found in the paper can be produced by running `cross_validate.py` with the appropriate arguments. Note that this may take several hours depending on the network, disease set, and computing setup. For the smallest network and disease set, run:

```
python cross_validate.py -n hprd -d ot --split_ratio 0.5 --runs 10
```

will run the cross-validation on the HPRD PPI network with the Open Targets data set using a train/test split of 50/50, with results being averaged over 10 runs. This will produce `out/ot-hprd-0.500.csv`, which can be used for further analysis, as well as plots in the `plots` directory.

### Predictions

Predictions can be made for any of the networks 

```
{"gmb", "wl", "biogrid", "string", "iid", "apid", "hprd"}
```

and any of the disease sets 

```
{"gmb", "dgn", "ot"}.
```

For a given a disease in the chosen disease set, the top N predictions of the quantum walk method can be calculated using `predictions.py`. For example:

```
python predictions.py --disease_set gmb --network wl --disease asthma --topn 200
```

will produce a `csv` file containing the top 200 predictions for the `asthma` seed genes contained in the `gmb` dataset, with the quantum walk being performed on the `wl` PPI network.

For a list of available diseases for a particular PPI network and disease set, you can run:

```
python avail_diseases.py -n biogrid -D dgn
```

as an example.

The time and diagonal hyperparameters can be set by modifying `predictions.py` accordingly.

Alternatively, custom disease datasets and/or PPI networks can be used by modifying the code in `qdgp/data.py`.
