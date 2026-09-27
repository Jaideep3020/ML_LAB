import os
import random
from pathlib import Path
import duckdb
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sklearn.model_selection import train_test_split
import torch

def frenchify(text):
    if not text: return text
    replacements = {
        ' Street': ' Rue', ' St': ' Rue', ' Road': ' Route', 
        ' Rd': ' Route', ' Avenue': ' Avenue', ' Ave': ' Avenue',
        ' LLC': ' SARL', ' Inc': ' SA', ' Ltd': ' SARL'
    }
    for eng, fra in replacements.items():
        text = text.replace(eng, fra).replace(eng.lower(), fra.lower()).replace(eng.upper(), fra.upper())
    return text

def main():
    run_dir = Path(r"C:\Users\jaide\Documents\Codex\2026-09-25\the\work\full")
    db_path = run_dir / 'cache' / 'data.duckdb'
    
    print("Extracting Massive Hard Negatives and Positives from DuckDB...")
    con = duckdb.connect()
    con.execute(f"ATTACH '{db_path}' AS db")
    
    # Increase to 350,000 samples to give the model way more data to learn from
    rows = con.execute(f"""
        WITH samples AS (
            SELECT qid, tid, label, score
            FROM read_parquet('{run_dir}/cache/oof-0.parquet')
            WHERE label = 1 OR (label = 0 AND score > 0.1)
            USING SAMPLE 350000 (Reservoir)
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
        text1 = f"{row[1] or ''} {row[2] or ''}".strip()
        text2 = f"{row[3] or ''} {row[4] or ''}".strip()
        
        # 15% chance to simulate French addresses to prepare the model for the Test set!
        if random.random() < 0.15:
            text1 = frenchify(text1)
            text2 = frenchify(text2)
            
        if text1 and text2:
            train_examples.append(InputExample(texts=[text1, text2], label=label))
            
    print(f"Extracted {len(train_examples)} pairs for Ultra-Fine-Tuning.")
    
    train_examples, val_examples = train_test_split(train_examples, test_size=0.05, random_state=42)
    
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(val_examples, name='ber-dev')

    print("Loading Pre-Trained Model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', num_labels=1, max_length=256, device=device)

    print("Starting Ultra Fine-Tuning (3 Epochs)...")
    out_model_path = run_dir / 'finetuned_cross_encoder_ultra'
    
    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=3,                     # Increased from 1 to 3
        evaluation_steps=1000,
        warmup_steps=300,
        output_path=str(out_model_path),
        show_progress_bar=True
    )
    print(f"Success! Ultra Fine-tuned model saved to {out_model_path}")

if __name__ == '__main__':
    main()
