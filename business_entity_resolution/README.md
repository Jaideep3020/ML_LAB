# Business Entity Resolution Pipeline

## Requirements

1. Python 3.9+
2. Required packages are listed in `requirements.txt`.
3. An environment with AVX2 support (for DuckDB) and optionally CUDA (for faster neural scoring).

To install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Pipeline

To regenerate both output files (`candidate_pairs.tsv` and `matching_results.tsv`) against the test dataset from a clean run, execute the following command:

```bash
python run_pipeline.py --test-dir dataset/test --out-dir output/
```

**Note:**
The `run_pipeline.py` script acts as the main entrypoint. It performs the following steps end-to-end:
1. Copies the test dataset into a localized run environment.
2. Uses DuckDB to build local search indices (BM25).
3. Retrieves initial candidate pairs using the BM25 system.
4. Generates dense text representations and performs FAISS dense retrieval to augment the candidate pool.
5. Scores the candidate pairs using the fine-tuned Cross-Encoder model.
6. Applies global constraints and optimized country-specific thresholds.
7. Outputs the final `candidate_pairs.tsv` and `matching_results.tsv` into the specified `--out-dir`.

## Output

The script generates two files in the output directory you specify:
- `candidate_pairs.tsv`: A comprehensive set of candidate entities considered for matching.
- `matching_results.tsv`: The final 1:1 mapped predictions, fully conforming to the submission constraints (no multi-matches, no self-matches, valid dataset targets).

No external APIs or services are called during inference. All necessary code and model weights are bundled within this submission directory.
