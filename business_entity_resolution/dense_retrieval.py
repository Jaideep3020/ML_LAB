"""
dense_retrieval.py
==================
Hybrid Dense Vector + BM25 retrieval to replace/augment the existing
SQLite FTS5 candidate generation pipeline.

ARCHITECTURE UNDERSTANDING (from retrieval.py):
  - Current pipeline: SQLite FTS5 per country-source index
  - query_limit = 600 candidates per entity (not top-k=10!)
  - neural_score.py then re-ranks these 600 and takes top-k=10
  - The cardinality collapse happens at the neural_score re-ranking stage
    NOT at the BM25 retrieval stage (BM25 already retrieves 600!)

WHAT THIS SCRIPT ADDS:
  Dense vectors find semantically similar records that share NO common
  keywords, e.g.:
    - "SBI" vs "State Bank of India"
    - "BofA" vs "Bank of America"
    - Abbreviated franchise names vs full names
  These are missed by FTS5 keyword search but caught by embedding similarity.

PIPELINE:
  1. Encode all Source 2 + Source 3 target records using a multilingual
     Bi-Encoder (paraphrase-multilingual-MiniLM-L12-v2, 384-dim)
  2. Build a FAISS HNSW index for fast ANN search
  3. Query every Source 1 test entity to get top-20 dense candidates
  4. Merge with existing BM25 candidates from the parquet files
  5. Write merged candidate_pairs to a new parquet for neural_score.py

TIME ESTIMATE on RTX 4060:
  Encoding ~5-6M records:    ~35 minutes
  FAISS HNSW index build:    ~20 minutes
  Querying 1.7M test entities: ~15 minutes
  Total:                     ~70 minutes

INSTALL REQUIREMENTS (run once):
  pip install faiss-gpu sentence-transformers
  # OR (CPU-only, slower):
  pip install faiss-cpu sentence-transformers
"""

import os
import gc
import json
import time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import duckdb

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--run-dir', required=True)
args = parser.parse_args()

# ── Configuration ──────────────────────────────────────────────────────────────
run_dir  = Path(args.run_dir)
db_path  = run_dir / 'cache' / 'data.duckdb'

# Output: merged candidate parquet for neural_score.py
DENSE_CANDIDATES_OUT = run_dir / 'cache' / 'dense_candidates_test.parquet'
MERGED_SCORES_OUT    = run_dir / 'cache' / 'neural_scores_test_hybrid.parquet'

# Bi-Encoder model: multilingual, handles English + French + Hindi
BI_ENCODER_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # 420MB, Apache 2.0

# FAISS config
FAISS_INDEX_PATH  = run_dir / 'cache' / 'faiss_target_index.bin'
FAISS_ID_MAP_PATH = run_dir / 'cache' / 'faiss_id_map.npy'   # maps FAISS int index → entity_id string

DENSE_TOP_K   = 20    # Dense candidates per query entity (retrieve more to cover k>5 entities)
ENCODE_BATCH  = 512   # Encoding batch size (tune down if OOM)
FAISS_NPROBE  = 64    # HNSW search width (higher = better recall, slower)

def format_text(name, address):
    """Concatenate name and address with separator for embedding."""
    n = (name or '').strip()
    a = (address or '').strip()
    if n and a:
        return f"{n} | {a}"
    return n or a

def encode_in_batches(model, texts, batch_size=512, desc="Encoding"):
    """GPU-batched encoding with progress."""
    all_embeddings = []
    total = len(texts)
    start = time.time()
    for i in range(0, total, batch_size):
        batch = texts[i:i+batch_size]
        embs = model.encode(batch, convert_to_numpy=True,
                            normalize_embeddings=True,  # for cosine via dot product
                            show_progress_bar=False)
        all_embeddings.append(embs)
        if (i // batch_size) % 20 == 0:
            elapsed = time.time() - start
            rate = (i + len(batch)) / max(elapsed, 1)
            print(f"  {desc}: {i+len(batch):,}/{total:,} ({rate:.0f} rec/s)")
    return np.vstack(all_embeddings).astype('float32')


def main():
    print("="*60)
    print("HYBRID DENSE + BM25 RETRIEVAL")
    print("="*60)

    # ── Check dependencies ─────────────────────────────────────────
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
        print(f"  faiss version: {faiss.__version__}")
        use_gpu_faiss = hasattr(faiss, 'StandardGpuResources')
        print(f"  FAISS GPU: {'YES' if use_gpu_faiss else 'NO (CPU mode)'}")
    except ImportError as e:
        print(f"\n[ERROR] Missing dependency: {e}")
        print("Install with:  pip install faiss-gpu sentence-transformers")
        print("Or CPU only:   pip install faiss-cpu sentence-transformers")
        return

    # ── Load DB ────────────────────────────────────────────────────
    con = duckdb.connect()
    con.execute(f"ATTACH '{db_path}' AS db")

    # ── Phase 1: Encode Targets (S2 + S3) ─────────────────────────
    if FAISS_INDEX_PATH.exists() and FAISS_ID_MAP_PATH.exists():
        print("\n[Phase 1] Loading cached FAISS index (already built)...")
        import faiss
        index = faiss.read_index(str(FAISS_INDEX_PATH))
        id_map = np.load(str(FAISS_ID_MAP_PATH), allow_pickle=True)
        print(f"  Index loaded: {index.ntotal:,} vectors")
    else:
        print("\n[Phase 1] Loading Bi-Encoder model...")
        from sentence_transformers import SentenceTransformer
        bi_encoder = SentenceTransformer(BI_ENCODER_MODEL)
        bi_encoder.max_seq_length = 128  # Keep fast; business names are short

        print("\n[Phase 2] Fetching all target records (S2 + S3)...")
        targets = con.execute("""
            SELECT entity_id, business_name, business_address FROM db.test_s2
            UNION ALL
            SELECT entity_id, business_name, business_address FROM db.test_s3
        """).fetchall()
        print(f"  Total targets: {len(targets):,}")

        entity_ids = np.array([r[0] for r in targets])
        texts = [format_text(r[1], r[2]) for r in targets]
        del targets; gc.collect()

        print("\n[Phase 3] Encoding targets on GPU...")
        embeddings = encode_in_batches(bi_encoder, texts, ENCODE_BATCH, "Target encoding")
        del texts; gc.collect()

        print(f"\n  Embedding shape: {embeddings.shape}")
        dim = embeddings.shape[1]

        print("\n[Phase 4] Building FAISS HNSW index...")
        import faiss
        # HNSW32: fast, good recall, no GPU training needed
        index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        index.add(embeddings)
        del embeddings; gc.collect()

        print(f"  Index built: {index.ntotal:,} vectors")
        faiss.write_index(index, str(FAISS_INDEX_PATH))
        np.save(str(FAISS_ID_MAP_PATH), entity_ids)
        print(f"  Saved index to {FAISS_INDEX_PATH}")

    # ── Phase 5: Encode Queries (S1) ──────────────────────────────
    print("\n[Phase 5] Fetching Source 1 test entities...")
    queries = con.execute("""
        SELECT entity_id, business_name, business_address, country
        FROM db.test_s1
        ORDER BY rid
    """).fetchall()
    print(f"  Total queries: {len(queries):,}")

    print("\n[Phase 6] Encoding queries...")
    from sentence_transformers import SentenceTransformer
    bi_encoder = SentenceTransformer(BI_ENCODER_MODEL)
    bi_encoder.max_seq_length = 128

    query_ids   = [r[0] for r in queries]
    query_texts = [format_text(r[1], r[2]) for r in queries]
    del queries; gc.collect()

    query_embs = encode_in_batches(bi_encoder, query_texts, ENCODE_BATCH, "Query encoding")
    del query_texts, bi_encoder; gc.collect()

    # ── Phase 7: FAISS Search ──────────────────────────────────────
    print(f"\n[Phase 7] Searching FAISS index (top-{DENSE_TOP_K} per query)...")
    index.hnsw.efSearch = FAISS_NPROBE
    start = time.time()

    SEARCH_BATCH = 10_000
    all_qids, all_tids, all_scores = [], [], []
    id_map = np.load(str(FAISS_ID_MAP_PATH), allow_pickle=True)

    for i in range(0, len(query_embs), SEARCH_BATCH):
        batch = query_embs[i:i+SEARCH_BATCH]
        scores, indices = index.search(batch, DENSE_TOP_K)
        for j, (sc_row, idx_row) in enumerate(zip(scores, indices)):
            qid = query_ids[i + j]
            for sc, idx in zip(sc_row, idx_row):
                if idx >= 0 and sc > 0.60:   # Cosine similarity threshold
                    all_qids.append(qid)
                    all_tids.append(str(id_map[idx]))
                    all_scores.append(float(sc))
        if i % 100_000 == 0:
            rate = (i + len(batch)) / max(time.time() - start, 1)
            print(f"  Searched: {i+len(batch):,}/{len(query_embs):,} ({rate:.0f} q/s)")

    del query_embs, index, id_map; gc.collect()

    print(f"\n  Dense candidates found: {len(all_qids):,}")
    dense_df = pa.table({
        'qid':          pa.array(all_qids,   type=pa.string()),
        'tid':          pa.array(all_tids,   type=pa.string()),
        'dense_score':  pa.array(all_scores, type=pa.float32()),
    })
    pq.write_table(dense_df, DENSE_CANDIDATES_OUT, compression='zstd')
    print(f"  Saved dense candidates to {DENSE_CANDIDATES_OUT}")

    # ── Phase 8: Merge with existing BM25 neural scores ───────────
    existing_scores = run_dir / 'cache' / 'neural_scores_test.parquet'
    if existing_scores.exists():
        print("\n[Phase 8] Merging Dense candidates with existing BM25 neural scores...")
        print("  (Dense candidates that are NOT in BM25 pool will need re-scoring)")
        merge_result = con.execute(f"""
            SELECT qid, tid,
                   CASE WHEN n.neural_score IS NOT NULL
                        THEN greatest(n.neural_score, d.dense_score)
                        ELSE d.dense_score
                   END AS neural_score
            FROM read_parquet('{DENSE_CANDIDATES_OUT}') d
            LEFT JOIN read_parquet('{existing_scores}') n USING (qid, tid)
            
            UNION
            
            SELECT qid, tid, neural_score
            FROM read_parquet('{existing_scores}')
        """).arrow()
        pq.write_table(merge_result, MERGED_SCORES_OUT, compression='zstd')
        print(f"  Merged output written to: {MERGED_SCORES_OUT}")
        print(f"  Total merged pairs: {len(merge_result):,}")
        print("\n  IMPORTANT: Dense candidates NOT in the BM25 pool have only")
        print("  their cosine similarity as score (no Cross-Encoder score).")
        print("  For best results, run neural_score.py on the dense candidates")
        print("  separately and re-merge.")
    else:
        print("\n[Phase 8] No existing BM25 neural scores found.")
        print("  Run neural_score.py on the dense candidates directly.")

    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    print(f"\nDense candidate file:   {DENSE_CANDIDATES_OUT}")
    print(f"Merged scores file:     {MERGED_SCORES_OUT}")
    print("\nNext steps:")
    print("  Option A (fast): Update apply_neural_thresholds.py to read from")
    print(f"    {MERGED_SCORES_OUT}")
    print("  Option B (best): Run neural_score.py on dense-only candidates,")
    print("    then re-merge for a fully Cross-Encoder-scored hybrid result.")

if __name__ == '__main__':
    main()
