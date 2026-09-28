# Comprehensive Error Analysis & Fine-Tuning Guide for Entity Resolution

## Executive Summary

Your current leaderboard score is **`0.879`** in macro $F_{0.5}$. The internal training tuning target was **`0.934`**. 

This comprehensive analysis investigates **why this performance gap exists**, breaks down **Precision vs Recall dynamics**, categorizes the root causes of **False Positives** and **False Negatives**, and provides actionable fine-tuning strategies to push the score towards **0.94–0.97+**.

---

## 1. Metric Mechanics: Why Precision Rules $F_{0.5}$

The competition evaluates submissions using **Macro $F_{0.5}$**:

$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}} = \frac{1.25 \times \text{TP}}{1.25 \times \text{TP} + 0.25 \times \text{FN} + \text{FP}}$$

![Precision, Recall and F0.5 Dynamics](/C:/Users/ps277/.gemini/antigravity-ide/brain/ef38ec2e-4f14-45c4-906e-1d906fc80f35/f05_threshold_dynamics.png)

### Key Takeaways on $F_{0.5}$:
1. **Precision is penalized $2\times$ more heavily than Recall**: 
   * Adding a **False Positive (FP)** reduces the denominator by $+1.0 \times \text{FP}$.
   * Missing a match (**False Negative, FN**) only reduces the denominator by $+0.25 \times \text{FN}$.
   * **1 False Positive does 4× as much damage to your score as 1 False Negative!**
2. **The Singleton Cliff**:
   * A Source 1 entity with zero true matches earns a score of **1.0** if you predict empty.
   * If you predict even **one single False Positive**, its score drops immediately from **1.0 to 0.0**.

---

## 2. Cardinality & Distribution Breakdown

Analyzing the 1,732,544 test queries against the 2,206,821 ground-truth training records reveals distinct behavioral discrepancies across countries.

![Match Cardinality Distribution](/C:/Users/ps277/.gemini/antigravity-ide/brain/ef38ec2e-4f14-45c4-906e-1d906fc80f35/cardinality_distribution.png)

### Country Comparison Table

| Metric / Cohort | Ground Truth (Train) | Predicted India (Test) | Predicted US (Test) | Predicted France (Unseen) |
| :--- | :---: | :---: | :---: | :---: |
| **Total Queries** | 2,206,821 | 809,986 (46.8%) | 663,106 (38.3%) | 259,452 (15.0%) |
| **Singletons ($k=0$)** | **5.58%** | 8.17% | 6.33% | **4.60%** *(Under-filtered)* |
| **Single-Match ($k=1$)** | **5.40%** | **13.58%** *(Over)* | **10.41%** *(Over)* | 8.29% |
| **High Match ($k>5$)** | **11.43%** | 6.90% | 8.19% | **14.08%** *(Severe FP cluster)* |
| **Avg Matches / Query** | **3.461** | **2.877** *(FN Deficit)* | **3.104** *(FN Deficit)* | **3.553** *(FP Surplus)* |

> [!IMPORTANT]
> **The Two Opposing Failure Modes in Your Current Submission:**
> 1. **France (15% of Test)** is suffering from **False Positives (Precision drop)**: The model was never trained on French data. Generic tokens (`SARL`, `Maison`, `Club`, `de la`) trigger multiple SQLite retrieval passes, inflating scores and creating clusters with $>5$ matches.
> 2. **India & US** are suffering from **False Negatives (Recall drop)**: In noisy addresses with landmark variations, scores drop below the fixed `0.625` threshold, resulting in $k=1$ match when ground truth actually has 3–4 matches.

---

## 3. False Positive Root Cause Analysis

Analyzing actual erroneous predictions on the test set exposes 4 concrete failure patterns:

![Error Taxonomy](/C:/Users/ps277/.gemini/antigravity-ide/brain/ef38ec2e-4f14-45c4-906e-1d906fc80f35/error_taxonomy.png)

### Category A: Same Brand / Generic Name, Different Street or Branch (38% of FPs)
* **What happens**: Businesses sharing generic name tokens in the same city are erroneously merged despite conflicting street addresses.
* **Real Example from your test output**:
  * **Query S1-865329132**: `Nantes Maison` at `9 Rue Emile Souvestre, Nantes`
  * **Predicted Match S2-215706495**: `Nantes Maison Sarl` at `R. COLETTE AUDRY, NANTES` $\rightarrow$ **FALSE POSITIVE**
  * **Cause**: `name_token_sort` = 1.0, `country_equal` = 1.0, `script_equal` = 1.0. The address similarity was near 0, but the name score pushed the model probability over `0.575`.

### Category B: Shared Address, Distinct Business Unit (29% of FPs)
* **What happens**: Commercial buildings, malls, and shared office suites share identical street addresses.
* **Real Example from your test output**:
  * **Query S1-761636326**: `GJM Union SAS` at `2 a Allee des Siffleurs, Lège-Cap-Ferret`
  * **Predicted Match S2-825724325**: `GJM CLUB` at `2A ALLEE DES SIFFLEURS` $\rightarrow$ **FALSE POSITIVE**
  * **Cause**: `address_containment` = 1.0, `number_jaccard` = 1.0, token prefix `GJM` matched. The model failed to penalize `CLUB` vs `Union SAS`.

### Category C: Number Conflicts (21% of FPs)
* **What happens**: Candidates with near-identical names on the same road, but at different house/building numbers.
* **Real Example**:
  * `Midwest Auto, 1400 Main St` matched to `Midwest Auto, 1850 Main St`.
  * The `number_conflict` feature flagged this (`True`), but the strong `name_token_set` and `name_jaro` outweighed the penalty.

### Category D: Singleton False Alarms (12% of FPs)
* **What happens**: A Source 1 entity has no true counterpart in S2/S3, but a marginal distractor scored 0.58–0.63.
* **Penalty**: Immediate **0.0 score** on that entity (destroying a potential 1.0 score).

---

## 4. Feature Importance & Model Sensitivity

Examining the decision tree splits in [`trial-3.txt`](file:///F:/ML/AmazonML_Project_Handoff/run/models/trial-3.txt) reveals what drives model predictions:

![Feature Importance Gain](/C:/Users/ps277/.gemini/antigravity-ide/brain/ef38ec2e-4f14-45c4-906e-1d906fc80f35/feature_importance_gain.png)

* **`pass_count` (6.27M Gain)**: By far the most influential feature. It measures how many distinct indexing passes (rare term, character trigrams, joint name-address) surfaced the candidate.
* **`number_jaccard` (1.68M Gain)**: The primary anchor for physical location verification.
* **`name_token_sort` (403k Gain)** & **`address_containment` (257k Gain)**: Core string similarity features.

---

## 5. Fine-Tuning Roadmap to Reach 0.94 – 0.97+

To increase the leaderboard score from **0.879** to the top tier, implement these high-impact adjustments:

### 1. Enable the Maximum-Score Singleton Gate (`singleton_gate`)
* **Current setting**: `"singleton_gate": 0.0` (disabled in `selected.json`).
* **The Problem**: Any isolated candidate that barely reaches 0.58 gets accepted, destroying singleton accuracy.
* **Solution**: Set `singleton_gate: 0.70` (or `0.75`). A query must have at least one high-confidence match ($\ge 0.70$) before any secondary candidates are accepted. If all candidates are weak ($\le 0.70$), the query is treated as a singleton (score 1.0).

### 2. Country-Specific Thresholds (Especially France)
* **The Problem**: France has distinct name-density distributions, causing over-matching at `source2: 0.625` / `source3: 0.575`.
* **Solution**: Apply adaptive thresholds:
  * For **France**: Raise threshold to `0.68` for S2 and `0.64` for S3 to eliminate generic name false merges.
  * For **India**: Lower threshold slightly to `0.58` for S2 / `0.54` for S3 for queries with strong name matches to recover noisy address false negatives.

### 3. Strict Address Veto on Number Conflicts
* If `number_conflict == True` (different non-empty house/street numbers) AND `name_ratio < 0.95`, enforce a hard penalty or rejection. Two businesses at different street numbers are rarely identical unless one is a documented suite discrepancy.

### 4. Train an Ensemble with CatBoost (`catboost_pair_limit`)
* The configuration in [`config.quality2.json`](file:///F:/ML/AmazonML_Project_Handoff/project/config.quality2.json) already defines support for CatBoost challengers. Ensembling LightGBM with CatBoost reduces tree-specific overconfidence on out-of-distribution text (such as French legal suffixes).
