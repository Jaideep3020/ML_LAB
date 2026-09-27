"""
Cross-encoder scoring script for existing candidate pairs.
Scores using mDeBERTa-v3-base cross-encoder.
"""
import argparse
import os
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--top-k', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()

    from sentence_transformers import CrossEncoder

    root = Path(args.run_dir).resolve()
    import sys; sys.path.insert(0, str(root.parent / 'src'))
    from ber.common import Context
    ctx = Context(args.run_dir, args.config)

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading cross-encoder model on {device.upper()}...")
    model = CrossEncoder(str(root / 'finetuned_cross_encoder_ultra'), max_length=256, device=device)

    db = ctx.db()
    
    # Read the candidate_pairs.tsv from the amazon_ml_result folder directly!
    candidates_tsv = r"C:\Users\jaide\Documents\Codex\2026-09-25\c\outputs\amazon_ml_result\candidate_pairs.tsv"
    
    print("Fetching candidates from TSV and joining text...")
    
    # We use DuckDB to unnest the comma-separated candidate IDs and join with the actual text.
    # We join with test_s1, test_s2, and test_s3 tables to get the business names and addresses.
    db.execute(f"""
        CREATE TEMP TABLE unnested_candidates AS
        SELECT source1_entity_id AS qid, unnest(string_split(candidate_entity_ids, ',')) AS tid
        FROM read_csv_auto('{candidates_tsv}', header=True)
        WHERE candidate_entity_ids != '';
    """)
    
    # We only take the top 1 candidate per query if needed, but since it's already filtered, 
    # we can just take the first one (order by tid as a tie-breaker).
    rows = db.execute(f"""
        SELECT c.qid, c.tid,
               q.business_name, q.business_address,
               r.business_name, r.business_address
        FROM unnested_candidates c
        JOIN test_s1 q ON q.entity_id = c.qid
        JOIN (
            SELECT entity_id, business_name, business_address FROM test_s2
            UNION ALL 
            SELECT entity_id, business_name, business_address FROM test_s3
        ) r ON r.entity_id = c.tid
        QUALIFY row_number() OVER(PARTITION BY c.qid ORDER BY c.tid) <= {args.top_k}
    """).fetchall()
    db.close()

    print(f"Scoring {len(rows)} pairs...")

    pairs = [(f"{r[2]} {r[3]}", f"{r[4]} {r[5]}") for r in rows]
    qids = [r[0] for r in rows]
    tids = [r[1] for r in rows]

    scores = []
    for i in range(0, len(pairs), args.batch_size):
        batch = pairs[i:i+args.batch_size]
        s = model.predict(batch, show_progress_bar=False)
        s = 1 / (1 + np.exp(-np.asarray(s, dtype=np.float32)))
        scores.extend(s.tolist())
        if (i // args.batch_size) % 50 == 0:
            print(f"  Scored {i+len(batch):,}/{len(pairs):,}")

    out = root / 'cache' / 'neural_scores_test.parquet'
    out.parent.mkdir(parents=True, exist_ok=True)
    
    table = pa.table({
        'qid': pa.array(qids, type=pa.string()),
        'tid': pa.array(tids, type=pa.string()),
        'neural_score': pa.array(scores, type=pa.float32()),
    })
    pq.write_table(table, out, compression='zstd')
    print(f"Saved neural scores -> {out}")

if __name__ == '__main__':
    main()
