import os
import random
from pathlib import Path
import duckdb
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sklearn.model_selection import train_test_split
import torch

def main():
    run_dir = Path(r"C:\Users\jaide\Documents\Codex\2026-09-25\the\work\full")
    db_path = run_dir / 'cache' / 'data.duckdb'
    oof_path = run_dir / 'cache' / 'oof-cat.parquet' # Or one of the oof-0/1/2.parquet files
    
    # Use one fold for fast sampling
    oof_path = run_dir / 'cache' / 'oof-0.parquet'
    
    print("Extracting Hard Negatives and Positives from DuckDB...")
    con = duckdb.connect()
    con.execute(f"ATTACH '{db_path}' AS db")
    
    # We fetch positive pairs and HARD negative pairs (where LightGBM gave it a high score but it was false)
    # qid is internal int32 (rid from train_s1), tid is string (entity_id from s2/s3)
    rows = con.execute(f"""
        WITH samples AS (
            SELECT qid, tid, label, score
            FROM read_parquet('{oof_path}')
            WHERE label = 1 OR (label = 0 AND score > 0.2)
            USING SAMPLE 150000 (Reservoir)
        )
        SELECT s.label, q.business_name, q.business_address, r.business_name, r.business_address
        FROM samples s
        JOIN db.train_s1 q ON q.rid = s.qid
        JOIN (
            SELECT entity_id, business_name, business_address FROM db.train_s2
            UNION ALL 
            SELECT entity_id, business_name, business_address FROM db.train_s3
        ) r ON r.entity_id = s.tid
    """).fetchall()
    
    train_examples = []
    for row in rows:
        label = float(row[0])
        # Format: "Name Address"
        text1 = f"{row[1] or ''} {row[2] or ''}".strip()
        text2 = f"{row[3] or ''} {row[4] or ''}".strip()
        if text1 and text2:
            train_examples.append(InputExample(texts=[text1, text2], label=label))
            
    print(f"Extracted {len(train_examples)} pairs for training.")
    
    # Split into train/val
    train_examples, val_examples = train_test_split(train_examples, test_size=0.1, random_state=42)
    
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
    
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(val_examples, name='ber-dev')

    print("Loading Pre-Trained Model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', num_labels=1, max_length=256, device=device)

    print("Starting Fine-Tuning...")
    out_model_path = run_dir / 'finetuned_cross_encoder'
    
    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=1,
        evaluation_steps=500,
        warmup_steps=100,
        output_path=str(out_model_path),
        show_progress_bar=True
    )
    print(f"Success! Fine-tuned model saved to {out_model_path}")

if __name__ == '__main__':
    main()
