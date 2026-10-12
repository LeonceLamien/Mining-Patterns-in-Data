<div align="center">

# Mining Patterns in Data

*Frequent itemset mining, sequential pattern discovery, and Bayesian networks*

![Python](https://img.shields.io/badge/Python-3-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-used-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-used-150458.svg)
![scikit--learn](https://img.shields.io/badge/scikit--learn-used-F7931E.svg)

[About](#about) • [Projects](#projects) • [Algorithms](#algorithms) • [Usage](#usage) • [Results](#results) • [Structure](#repository-structure)

</div>

---

## About

This repository brings together three data-mining projects completed as part of the **Mining pattern** coursework. It explores how patterns can be discovered in transactional and sequential data, and how probabilistic graphical models can represent and reason about uncertain data.

The implementations cover:

- Frequent itemset mining with **Apriori** and a vertical-database depth-first miner.
- Sequential pattern mining with **SPADE**, including a supervised ranking based on **Weighted Relative Accuracy (WRAcc)**.
- **Bayesian networks**: parameter learning, inference for missing values, and structure learning.

The repository includes sample datasets, reference/learned outputs, and notebooks for performance and results analysis.

---

## Projects

### 1. Frequent Itemset Mining

`project1/p1-apriori/` contains two miners for transaction databases:

- **Apriori**: generates candidate itemsets level by level and counts their support.
- **Vertical depth-first miner**: represents each item by the transactions containing it, then intersects transaction-ID sets while extending patterns.

Both algorithms use a minimum-frequency threshold and return frequent itemsets in a trie that can be displayed with `print_trie`. Included datasets cover examples such as `toy`, `retail`, `mushroom`, and `accidents`.

> `project1/frequent_itemset_miner.py` is the original coursework skeleton. Use the implementation in `project1/p1-apriori/frequent_itemset_miner.py`.

### 2. Sequential Pattern Mining and Classification

`Project2/` mines ordered sequences from positive and negative sequence datasets:

- **SPADE-style mining** uses a vertical occurrence representation and depth-first pattern extension.
- **Top-k frequent sequences** are ranked by total support.
- **Supervised sequence mining** ranks patterns by WRAcc to highlight patterns that distinguish positive from negative examples.
- **Decision-tree classifiers** use mined patterns as binary features and report accuracy under k-fold cross-validation.

The datasets include protein sequences and small Reuters collections. The `for_analysis/` notebooks and scripts support comparisons of mining and classification variants.

### 3. Bayesian Networks

`project3/` implements discrete Bayesian networks and works with **BIF** network files and CSV datasets. The code includes:

- Bayesian-network parsing and BIF output.
- Conditional probability table learning with Dirichlet smoothing.
- Exact inference for one or two missing variables.
- Structure-learning experiments: a greedy likelihood-based approach in `bayesiannetwork.py` and a **Chow–Liu maximum-spanning-tree** approach in `bayesiannetwork2.py`.

Datasets are provided for networks including Asia, Alarm, Sachs, Sprinkler, Andes, and Water. Learned BIF files and a results notebook are also included.

---

## Algorithms

### Frequent Itemsets

Apriori exploits the property that every subset of a frequent itemset must also be frequent. The alternative miner stores each item's transaction IDs and uses set intersections to grow patterns depth-first.

### Sequential Patterns

Sequences are stored as item occurrences indexed by sequence and position. Extending a pattern checks whether the next item occurs later in the same sequence. The supervised miner scores patterns with WRAcc before selecting features for a decision tree.

### Bayesian Networks

The network code represents each variable with a conditional probability table. Inference evaluates candidate values against observed evidence; structure-learning variants estimate dependencies from training data and then fit the resulting tables.

---

## Usage

### Requirements

- Python 3
- NumPy
- pandas
- scikit-learn (needed for the Project 2 classifiers)

Install the dependencies:

```bash
python -m pip install numpy pandas scikit-learn
```

There is currently no `requirements.txt` or shared command-line interface; each coursework project has its own entry points and data format.

### Project 1: Run the Itemset Miners

The miners are Python functions. From the implementation directory, run both against the toy dataset and display their itemset tries:

```bash
cd project1/p1-apriori
python - <<'PY'
from frequent_itemset_miner import apriori, alternative_miner, print_trie

dataset = "Datasets/toy/toy.dat"
min_frequency = 0.25

print("Apriori")
print_trie(apriori(dataset, min_frequency))

print("Vertical depth-first miner")
print_trie(alternative_miner(dataset, min_frequency))
PY
```

### Project 2: Mine Sequences or Evaluate a Classifier

Run these commands from `Project2/`. Mining commands take positive and negative sequence files followed by `k`.

```bash
python frequent_miner.py \
  datasets/Protein/SRC1521.txt \
  datasets/Protein/PKA_group15.txt \
  10

python supervised_miner.py \
  datasets/Protein/SRC1521.txt \
  datasets/Protein/PKA_group15.txt \
  10
```

To evaluate a pattern-based classifier, provide the minimum frequency and number of cross-validation folds as the last two arguments:

```bash
python frequent_classifier.py \
  datasets/Reuters_small/acq_small.txt \
  datasets/Reuters_small/earn_small.txt \
  10 0.05 5

python supervised_classifier.py \
  datasets/Reuters_small/acq_small.txt \
  datasets/Reuters_small/earn_small.txt \
  10 0.05 5
```

The classifier arguments are `positive_file negative_file k min_frequency number_of_folds`. `frequent_classifier.py` uses support-ranked patterns; `supervised_classifier.py` uses WRAcc-ranked patterns.

### Project 3: Learn a Bayesian Network Structure

Run the Chow–Liu learner from `project3/`. It reads the training data and a BIF network template, then writes the learned network:

```bash
cd project3
python bayesiannetwork2.py \
  datasets/asia/train.csv \
  datasets/asia/test.csv \
  datasets/asia/asia_missing_probabilities.bif \
  /tmp/learned_asia.bif
```

The script accepts the test CSV path for compatibility, but the current structure-learning routine does not use it. The network inference and parameter-learning methods are available from the `BayesianNetwork` class in `bayesiannetwork.py`.

---

## Results

The repository contains example/reference results rather than a single cross-project benchmark:

- **Project 1:** transaction datasets, frequent-itemset solution files, execution-time CSVs, and a performance-analysis notebook.
- **Project 2:** positive/negative sequence datasets, test outputs, and analysis notebooks for mining and classification variants.
- **Project 3:** complete and incomplete-probability BIF networks, learned network files, train/test CSVs, and a results notebook.

The notebooks document the experiments; measured values depend on the selected dataset, thresholds, and parameters.

---

## Repository Structure

```text
Mining-Patterns-in-Data/
├── project1/
│   ├── p1-apriori/
│   │   ├── frequent_itemset_miner.py   # Apriori and vertical DFS miners
│   │   ├── Datasets/                   # Transaction databases
│   │   ├── solutions/                  # Reference itemset outputs
│   │   ├── execution_times*.csv        # Runtime measurements
│   │   └── perf_analysis.ipynb          # Performance analysis
│   ├── solutions/                      # Additional reference outputs
│   └── frequent_itemset_miner.py       # Original coursework skeleton
├── Project2/
│   ├── frequent_miner.py               # Support-ranked sequential patterns
│   ├── supervised_miner.py             # WRAcc-ranked sequential patterns
│   ├── frequent_classifier.py          # Frequent-pattern classifier
│   ├── supervised_classifier.py        # WRAcc-pattern classifier
│   ├── datasets/                       # Protein, Reuters, and test sequences
│   └── for_analysis/                   # Experimental variants and notebooks
└── project3/
    ├── bayesiannetwork.py              # Bayesian-network inference/learning
    ├── bayesiannetwork2.py             # Chow–Liu structure-learning variant
    ├── datasets/                       # BIF networks and CSV data
    ├── result_bayesiannetwork2/        # Learned BIF networks
    └── results.ipynb                   # Results analysis
```

---

## Academic Context

Developed for **Mining Patterns in Data**. The repository contains coursework implementations and associated datasets and experimental outputs.

## License

No open-source license is currently specified in this repository.