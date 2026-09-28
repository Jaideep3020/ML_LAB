# Submission Results

## Leaderboard Scores

| Version | Score (F0.5) | Key Change |
|---------|-------------|-----------|
| Baseline (raw pipeline) | 0.625 | Un-finetuned cross-encoder, top-k=1 |
| v1 (1-epoch finetune) | 0.879 | 48,840 hard negative pairs, 1 epoch |
| **v2 (ultra finetune + country-aware)** | **0.935815** | **180k pairs, 3 epochs, French augmentation, country-aware thresholds** |

## v2 Submission Details

- **File:** `matching_results_neural.tsv`
- **Score:** `0.935815` (macro F0.5)
- **Improvement over v1:** +0.056815 (+6.4%)
- **Improvement over baseline:** +0.310815 (+49.7%)

## v2 Pipeline Configuration

### Fine-Tuning
- **Model:** `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
- **Training pairs:** 180,112 (hard negatives + positives)
- **Epochs:** 3
- **Synthetic French augmentation:** 15% of English pairs converted
  (Street→Rue, Road→Route, LLC→SARL, Avenue→Avenue)
- **Validation accuracy:** 99.15% | **Precision:** 97.05%

### Post-Processing (apply_neural_thresholds.py)
- **Country-aware thresholds:**
  - India: 0.48 (largest FN drift: -0.506 avg matches)
  - US: 0.52 (moderate FN drift: -0.336 avg matches)
  - France: 0.55 (OOD country, conservative)
- **Name similarity guard:** Jaro-Winkler >= 0.55
- **Global Target Ownership:** `row_number() OVER (PARTITION BY tid ORDER BY neural_score DESC)`

### Retrieval
- **BM25 FTS5:** SQLite per country-source index, query_limit=600
- **Neural re-ranking:** top-k=10 per entity

## Analysis Findings (from extensive_test_analysis.py)

| Metric | Value |
|--------|-------|
| Total scored pairs | 6,024,560 |
| Score median | 0.9999 (bimodal distribution) |
| Singleton rate (test) | 6.87% |
| Country violations | 0.000% |
| Probable FPs (name_sim < 0.70) | 593,549 (11.24%) |
| Proxy Precision | 88.76% |

## Remaining Gap Analysis (0.935 → 0.989 target)

| Root Cause | Impact | Fix |
|-----------|--------|-----|
| k=1 inflation: India 11.3% vs GT 5.4% | HIGH | Country thresholds ✅ done |
| k>5 deficit: ALL 8% vs GT 11.4% | HIGH | Raise top-k to 20 (5h re-inference) |
| Structural FPs (addr match, name mismatch) | MEDIUM | finetune_structural_hardneg.py |
| Abbreviation misses (SBI vs State Bank) | LOW | dense_retrieval.py (FAISS hybrid) |
