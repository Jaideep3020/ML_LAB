import argparse
import os
import shutil
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="End-to-end inference pipeline")
    parser.add_argument('--test-dir', required=True, help="Path to test data containing test_source1.tsv, etc.")
    parser.add_argument('--out-dir', required=True, help="Path to output directory")
    parser.add_argument('--run-dir', default="run", help="Path to run directory")
    args = parser.parse_args()

    test_dir = Path(args.test_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    expected_test_dir = run_dir / "data" / "dataset" / "test"
    if not expected_test_dir.exists():
        expected_test_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(test_dir, expected_test_dir)

    db_path = run_dir / "cache" / "data.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    import duckdb
    db = duckdb.connect(str(db_path))
    
    for source in (1, 2, 3):
        table = f"test_s{source}"
        tsv_path = expected_test_dir / f"test_source{source}.tsv"
        exists = db.execute("SELECT count(*) FROM information_schema.tables WHERE table_name=?", [table]).fetchone()[0]
        if exists:
            db.execute(f"DROP TABLE {table}")
        
        db.execute(f"""CREATE TABLE {table} AS
            SELECT row_number() OVER (ORDER BY entity_id)::INTEGER AS rid,
                entity_id, coalesce(business_name,'') AS business_name,
                coalesce(business_address,'') AS business_address,
                coalesce(country,'') AS country
            FROM read_csv('{tsv_path}', delim='\\t', header=true, all_varchar=true,
                            quote='', null_padding=false, strict_mode=true)""")
    db.close()

    print("Building indexes for BM25...")
    subprocess.run(["python", "-m", "src.ber.cli", "--run-dir", str(run_dir), "index", "--dataset", "test"], check=True)
    print("Retrieving BM25 candidates...")
    subprocess.run(["python", "-m", "src.ber.cli", "--run-dir", str(run_dir), "retrieve", "--dataset", "test"], check=True)
    
    print("Generating candidate_pairs.tsv from BM25 pipeline...")
    subprocess.run(["python", "-m", "src.ber.cli", "--run-dir", str(run_dir), "predict", "--dataset", "test", "--baseline"], check=True)
    bm25_candidate_tsv = run_dir / "output" / "candidate_pairs.tsv"

    print("Running neural_score.py...")
    subprocess.run([
        "python", "neural_score.py", 
        "--run-dir", str(run_dir), 
        "--config", str(Path("config.json").resolve()), 
        "--tag", "main", 
        "--candidate-tsv", str(bm25_candidate_tsv),
        "--model-dir", str((run_dir / "finetuned_cross_encoder_ultra").resolve())
    ], check=True)

    print("Running dense_retrieval.py...")
    subprocess.run([
        "python", "dense_retrieval.py", 
        "--run-dir", str(run_dir)
    ], check=True)

    print("Applying neural thresholds...")
    subprocess.run([
        "python", "apply_neural_thresholds.py", 
        "--run-dir", str(run_dir), 
        "--out-dir", str(out_dir),
        "--scores-path", str(run_dir / "cache" / "neural_scores_test_hybrid.parquet")
    ], check=True)

    print("Pipeline complete. Output files in", out_dir)

if __name__ == '__main__':
    main()
