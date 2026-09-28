# Business Entity Resolution System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Leaderboard F0.5 Score](https://img.shields.io/badge/Macro%20F0.5-0.935815-success.svg)](business_entity_resolution/RESULTS.md)

An end-to-end, high-performance machine learning system for cross-source **Business Entity Resolution**. Built to identify, match, and resolve noisy multi-source records (covering multiple jurisdictions: India, US, France) into unified business entities under strict 1:1 matching constraints.

---

## 📌 Project Overview

Entity resolution across distributed databases is often challenged by typographical noise, divergent abbreviation standards (e.g., "SBI" vs. "State Bank of India"), regional terminology discrepancies (e.g., French legal corporate structures like "SARL" vs. English "LLC"), and high combinatorial complexity.

This repository implements a multi-stage resolution pipeline:
1. **Candidate Retrieval (Sparse + Dense)**: High-speed candidate generation using DuckDB / SQLite FTS5 (BM25) combined with FAISS dense vector retrieval to ensure >99% recall while reducing pair comparisons from billions to manageable candidate sets.
2. **Neural Re-Ranking (Cross-Encoder)**: Deep semantic scoring using a fine-tuned multilingual cross-encoder (`mmarco-mMiniLMv2-L12-H384-v1`) trained on 180k+ hard negative pairs with synthetic language augmentation.
3. **Country-Aware Post-Processing**: Country-specific decision thresholds, Jaro-Winkler string similarity guards, and global target ownership constraints to eliminate duplicate target claims and false positives.

---

## 🏆 Key Benchmark Results

Our model progression achieved a final macro $F_{0.5}$ score of **0.935815** (+49.7% over baseline):

| Version | Macro $F_{0.5}$ Score | Key Innovations & Methodologies |
| :--- | :---: | :--- |
| **Baseline** | 0.625000 | Raw BM25 + un-finetuned cross-encoder, top-$k=1$ |
| **v1 (1-Epoch)** | 0.879000 | 48,840 hard negative pairs, 1-epoch fine-tuning |
| **v2 (Ultra + Country-Aware)** | **0.935815** | 180k pairs, 3 epochs, French text augmentation, country-specific thresholds, Jaro-Winkler guards |

Detailed analysis and error breakdown are documented in [RESULTS.md](business_entity_resolution/RESULTS.md) and [error_analysis_report.md](error_analysis_report.md).

---

## 📂 Repository Architecture

```text
ML_LAB/
├── README.md                                  # Repository entry point & overview (this file)
├── INSTRUCTIONS.md                            # Quickstart & system execution guide
├── error_analysis_report.md                   # In-depth error analysis, precision/recall audits
├── final_submission_execution_plan_1.md       # Technical execution plan & architecture spec
├── .gitignore                                 # Git ignore rules (weights, caches, virtualenvs)
│
└── business_entity_resolution/                # Core solution package
    ├── requirements.txt                       # Python dependencies
    ├── config.json                            # Global pipeline configuration
    ├── run_pipeline.py                        # Main end-to-end execution pipeline
    ├── README.md                              # Module-specific documentation
    ├── RESULTS.md                             # Benchmark and evaluation log
    ├── check_results.py                       # Validates output schemas & matching counts
    ├── analyze_validation.py                  # Validation metrics & F0.5 evaluator
    ├── apply_neural_thresholds.py             # Applies country-specific thresholds & 1:1 constraints
    ├── cardinality_breakdown.py               # Generates cardinality & match distribution statistics
    ├── dense_retrieval.py                     # FAISS-based dense semantic retriever
    ├── extensive_test_analysis.py             # Full diagnostics and confidence interval analysis
    ├── finetune_cross_encoder.py              # Baseline cross-encoder fine-tuning script
    ├── finetune_cross_encoder_ultra.py        # 3-epoch ultra fine-tuning with augmentation
    ├── finetune_structural_hardneg.py         # Structural hard negative generation & training
    ├── neural_score.py                        # Standalone neural inference & batch scorer
    ├── plot_cardinality_chart.py              # Visualizes match cardinalities across countries
    │
    └── ber/                                   # Internal library modules
        ├── __init__.py
        ├── background.py                      # Asynchronous tasks & background workers
        ├── baseline.py                        # Baseline matching heuristics
        ├── bounded_run.py                     # Resource-bounded execution controller
        ├── cli.py                             # Command-line interface definitions
        ├── common.py                          # Utility helpers, logging & I/O
        ├── data.py                            # Data loading, normalization & parsing
        ├── decisions.py                       # Final decision boundaries & 1:1 resolution logic
        ├── delivery.py                        # Submission formatter & artifact generator
        ├── distributed.py                     # Multi-process & parallel worker orchestration
        ├── experiments.py                     # Experiment tracking & trial management
        ├── features.py                        # String similarity, token, & structural feature extraction
        ├── metrics.py                         # Evaluation metrics (F0.5, Precision, Recall)
        ├── model.py                           # Cross-encoder model wrappers & inference pipeline
        ├── retrieval.py                       # BM25 & indexing algorithms
        ├── sample.py                          # Dataset stratification & sampling
        └── text.py                            # Text normalization & tokenization
```

---

## 🚀 Getting Started on Any System

Follow these steps to set up and run the entity resolution pipeline on a fresh machine (Linux, macOS, or Windows).

### 1. Prerequisites
- **Python**: Version 3.9 or higher
- **RAM**: Minimum 8 GB recommended (16 GB for full multi-source processing)
- **Disk Space**: At least 5 GB free disk space
- **GPU (Optional)**: NVIDIA GPU with CUDA for accelerated neural re-ranking (CPU inference is supported via PyTorch/FAISS-CPU)

### 2. Clone the Repository
```bash
git clone https://github.com/Jaideep3020/ML_LAB.git
cd ML_LAB
```

### 3. Set Up a Virtual Environment
```bash
# Create environment
python -m venv .venv

# Activate environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Windows (cmd.exe):
.venv\Scripts\activate.bat
```

### 4. Install Dependencies
All necessary libraries are listed in `business_entity_resolution/requirements.txt`:
```bash
pip install --upgrade pip
pip install -r business_entity_resolution/requirements.txt
```

#### Core Dependencies Included:
- `duckdb>=1.1.0`: High-performance analytical query engine for fast FTS5 and candidate indexing
- `torch>=2.0.0`: PyTorch for neural network inference
- `sentence-transformers>=2.0.0`: Cross-Encoder and Transformer model inference
- `faiss-cpu>=1.8.0`: High-dimensional vector indexing and dense retrieval
- `pandas>=2.0.0` & `numpy>=1.26.0`: Data manipulation and numerical operations
- `pyarrow>=17.0.0`: Efficient columnar data serialization

---

## ⚡ Execution & Usage

Navigate to the project directory:
```bash
cd business_entity_resolution
```

### 1. Run the Full End-to-End Pipeline
To run candidate generation, neural re-ranking, and post-processing on test datasets:
```bash
python run_pipeline.py --test-dir dataset/test --out-dir output/
```

The script will automatically:
- Ingest and index source records.
- Generate candidate matches with BM25 + dense retrieval.
- Score candidate pairs with the fine-tuned neural model.
- Apply country-aware thresholds (India: 0.48, US: 0.52, France: 0.55) and Jaro-Winkler guards ($\ge 0.55$).
- Enforce 1:1 match constraints and produce final deliverables.

### 2. Verify Output Files
Confirm that the output files have been generated:
```bash
python check_results.py
```
Outputs produced:
- `output/candidate_pairs.tsv`: All retrieved candidate pairs.
- `output/matching_results.tsv`: Clean, disambiguated 1:1 entity resolution pairings.

### 3. Evaluate & Analyze Diagnostics
Run validation checks and distribution analysis:
```bash
python analyze_validation.py
python cardinality_breakdown.py
```

---

## 🛠️ Configuration Options

Parameters can be adjusted in [`business_entity_resolution/config.json`](business_entity_resolution/config.json):
- `memory_limit`: Memory threshold allocated to DuckDB (e.g., `"8GB"`).
- `threads`: Number of worker threads for parallel indexing and scoring.
- `batch_size`: Batch size for neural inference.
- `candidate_budgets`: Candidate pool limits per retrieval pass (`[20, 50, 100]`).
- `retrieval_recall_target`: Target retrieval recall floor (default: `0.99`).

---

## 📄 License
This repository is open-source and available under the [MIT License](LICENSE).
