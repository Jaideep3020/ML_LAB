"""
finetune_structural_hardneg.py  (v2 - Critically Revised)
==========================================================
CHANGES FROM v1:
  [FIX 1] CROSS JOIN replaced with OOF-candidate-pool filter to avoid OOM
  [FIX 2] Class ratio corrected to 80% negative / 20% positive for F0.5
  [FIX 3] SEP token between name and address for better field weighting
  [FIX 4] Learning rate lowered 2e-5 -> 5e-6 to prevent catastrophic forgetting
  [FIX 5] Country-stratified positives to fix India k=1 Recall inflation

CRITICAL ANALYSIS OF ORIGINAL SCRIPT:
  1. CROSS JOIN train_s1 x all_targets = 2.2M x 6M = 13 TRILLION rows -> OOM crash
  2. 67/33 neg/pos ratio makes model too aggressive -> worsens k=1 inflation
  3. Plain text concat "name address" loses field boundary signal
  4. Default lr=2e-5 on warm-started weights -> catastrophic forgetting risk
  5. No country stratification -> India positives underrepresented vs its 1.82% drift
"""

import random
from pathlib import Path
import duckdb
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sklearn.model_selection import train_test_split
import torch

# Configuration
run_dir    = Path(r"C:\Users\jaide\Documents\Codex\2026-09-25\the\work\full")
db_path    = run_dir / 'cache' / 'data.duckdb'

BASE_MODEL = str(run_dir / 'finetuned_cross_encoder_ultra')
OUT_MODEL  = str(run_dir / 'finetuned_cross_encoder_structural')

# [FIX 2] 80/20 neg/pos ratio for F0.5 precision focus
N_STRUCTURAL_NEGATIVES = 70_000
N_STANDARD_NEGATIVES   = 30_000
N_POSITIVES            = 25_000

# [FIX 4] Low LR for continued fine-tuning (prevents catastrophic forgetting)
LEARNING_RATE = 5e-6
EPOCHS        = 2
BATCH_SIZE    = 32
MAX_LENGTH    = 256
EVAL_STEPS    = 400
WARMUP_STEPS  = 100

# [FIX 3] Separator between name and address fields
SEP = " | "

def format_record(name, address):
    n = (name or '').strip()
    a = (address or '').strip()
    if n and a:
        return f"{n}{SEP}{a}"
    return (n or a).strip()

# Connect
print("="*60)
print("STRUCTURAL HARD NEGATIVE FINE-TUNING v2 (Critically Revised)")
print("="*60)
print(f"  Base:   {BASE_MODEL}")
print(f"  Output: {OUT_MODEL}")
print(f"  LR: {LEARNING_RATE} | Epochs: {EPOCHS} | Batch: {BATCH_SIZE}")

con = duckdb.connect()
con.execute(f"ATTACH '{db_path}' AS db")

# Step 1: Structural Hard Negatives
# [FIX 1] Filter the existing OOF candidate pool (not CROSS JOIN)
print(f"\n[1/4] Mining {N_STRUCTURAL_NEGATIVES:,} structural hard negatives...")
print("      Filtering OOF pool: high address_sim + low name_sim")

structural_negs = con.execute(f"""
    WITH base AS (
        SELECT qid, tid
        FROM read_parquet('{run_dir}/cache/oof-0.parquet')
        WHERE label = 0 AND score > 0.05
        USING SAMPLE 300000 (Reservoir)
    ),
    targets AS (
        SELECT entity_id, business_name, business_address FROM db.train_s2
        UNION ALL
        SELECT entity_id, business_name, business_address FROM db.train_s3
    )
    SELECT
        s1.business_name, s1.business_address,
        t.business_name,  t.business_address
    FROM base b
    JOIN db.train_s1 s1 ON s1.rid = b.qid
    JOIN targets t ON t.entity_id = b.tid
    WHERE
        jaro_winkler_similarity(
            lower(coalesce(s1.business_address, '')),
            lower(coalesce(t.business_address,  ''))
        ) >= 0.80
        AND jaro_winkler_similarity(
            lower(coalesce(s1.business_name, '')),
            lower(coalesce(t.business_name,  ''))
        ) < 0.60
    LIMIT {N_STRUCTURAL_NEGATIVES}
""").fetchall()
print(f"      Retrieved: {len(structural_negs):,}")

# Step 2: Standard Hard Negatives
print(f"\n[2/4] Mining {N_STANDARD_NEGATIVES:,} standard hard negatives...")
standard_negs = con.execute(f"""
    WITH samples AS (
        SELECT qid, tid
        FROM read_parquet('{run_dir}/cache/oof-0.parquet')
        WHERE label = 0 AND score > 0.15
        USING SAMPLE {N_STANDARD_NEGATIVES} (Reservoir)
    )
    SELECT s1.business_name, s1.business_address, t.business_name, t.business_address
    FROM samples s
    JOIN db.train_s1 s1 ON s1.rid = s.qid
    JOIN (
        SELECT entity_id, business_name, business_address FROM db.train_s2
        UNION ALL
        SELECT entity_id, business_name, business_address FROM db.train_s3
    ) t ON t.entity_id = s.tid
""").fetchall()
print(f"      Retrieved: {len(standard_negs):,}")

# Step 3: Country-Stratified Positives
# [FIX 5] Equal sampling from India and US fixes India k=1 Recall inflation
print(f"\n[3/4] Mining {N_POSITIVES:,} country-stratified positives...")
per_country = N_POSITIVES // 2
positives = con.execute(f"""
    WITH india_samples AS (
        SELECT o.qid, o.tid
        FROM read_parquet('{run_dir}/cache/oof-0.parquet') o
        JOIN db.train_s1 s1 ON s1.rid = o.qid
        WHERE o.label = 1 AND s1.country = 'India'
        USING SAMPLE {per_country} (Reservoir)
    ),
    us_samples AS (
        SELECT o.qid, o.tid
        FROM read_parquet('{run_dir}/cache/oof-0.parquet') o
        JOIN db.train_s1 s1 ON s1.rid = o.qid
        WHERE o.label = 1 AND s1.country = 'US'
        USING SAMPLE {per_country} (Reservoir)
    ),
    combined AS (SELECT * FROM india_samples UNION ALL SELECT * FROM us_samples),
    targets AS (
        SELECT entity_id, business_name, business_address FROM db.train_s2
        UNION ALL
        SELECT entity_id, business_name, business_address FROM db.train_s3
    )
    SELECT s1.business_name, s1.business_address, t.business_name, t.business_address
    FROM combined c
    JOIN db.train_s1 s1 ON s1.rid = c.qid
    JOIN targets t ON t.entity_id = c.tid
""").fetchall()
print(f"      Retrieved: {len(positives):,}")

# Step 4: Assemble
print("\n[4/4] Building InputExamples with SEP tokens...")
examples = []

for row in structural_negs:
    t1 = format_record(row[0], row[1])
    t2 = format_record(row[2], row[3])
    if t1 and t2:
        examples.append(InputExample(texts=[t1, t2], label=0.0))

for row in standard_negs:
    t1 = format_record(row[0], row[1])
    t2 = format_record(row[2], row[3])
    if t1 and t2:
        examples.append(InputExample(texts=[t1, t2], label=0.0))

for row in positives:
    t1 = format_record(row[0], row[1])
    t2 = format_record(row[2], row[3])
    if t1 and t2:
        examples.append(InputExample(texts=[t1, t2], label=1.0))

random.shuffle(examples)
train_examples, val_examples = train_test_split(examples, test_size=0.05, random_state=42)

neg_count = sum(1 for e in examples if e.label == 0.0)
pos_count = sum(1 for e in examples if e.label == 1.0)
print(f"\n  Total:     {len(examples):,}")
print(f"  Negatives: {neg_count:,} ({neg_count/len(examples)*100:.1f}%) <- F0.5 precision focus")
print(f"  Positives: {pos_count:,} ({pos_count/len(examples)*100:.1f}%) <- Recall guard")
print(f"  Train/Val: {len(train_examples):,} / {len(val_examples):,}")

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
evaluator = CEBinaryClassificationEvaluator.from_input_examples(
    val_examples, name='structural-dev'
)

# Fine-Tune
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nLoading base model on {device.upper()}...")
model = CrossEncoder(BASE_MODEL, num_labels=1, max_length=MAX_LENGTH, device=device)

print(f"\nFine-tuning {EPOCHS} epoch(s) at lr={LEARNING_RATE}...")
print("  Monitor: Precision should rise; Recall must stay above 0.90\n")

model.fit(
    train_dataloader=train_dataloader,
    evaluator=evaluator,
    epochs=EPOCHS,
    evaluation_steps=EVAL_STEPS,
    warmup_steps=WARMUP_STEPS,
    output_path=OUT_MODEL,
    show_progress_bar=True,
    optimizer_params={'lr': LEARNING_RATE},
)

print(f"\n[DONE] Saved to: {OUT_MODEL}")
print("\nNext steps:")
print("  1. python neural_score.py ... --top-k 10  (~5h GPU)")
print("  2. python apply_neural_thresholds.py ...  (country-aware)")
