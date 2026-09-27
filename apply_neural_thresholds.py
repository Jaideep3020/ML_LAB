import argparse
from pathlib import Path
import duckdb

# ─────────────────────────────────────────────────────────────────
# Country-Aware Threshold Configuration
# Derived from cardinality drift analysis (GT vs Predicted):
#   India: singleton drift +1.82%, avg match drift -0.506  → most under-predicted
#   US:    singleton drift +0.84%, avg match drift -0.336  → moderately under-predicted
#   France: OOD country, no GT baseline → conservative
#   Default: safe fallback for any unseen country
# ─────────────────────────────────────────────────────────────────
COUNTRY_THRESHOLDS = {
    'India':  0.48,   # Most aggressive: biggest FN drift observed
    'US':     0.52,   # Moderate: smaller drift, still recovers collapsed k=1 entities
    'France': 0.55,   # Conservative: OOD country, no ground truth to calibrate against
}
DEFAULT_THRESHOLD = 0.55  # Safe fallback for any future unknown country

# ─────────────────────────────────────────────────────────────────
# Name Similarity Guard
# Pairs with low name similarity are likely FPs (same address, different business).
# A dual-gate: neural score AND name similarity must both pass.
# Set to 0.0 to disable the name guard entirely.
# ─────────────────────────────────────────────────────────────────
NAME_SIM_GUARD = 0.55   # Jaro-Winkler threshold; 0.0 = disabled


def main():
    parser = argparse.ArgumentParser(
        description="Apply country-aware neural thresholds to produce final matching_results.tsv"
    )
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    # Global threshold is kept as a fallback override (optional)
    parser.add_argument('--threshold', type=float, default=None,
                        help='Override all country-specific thresholds with a single value')
    args = parser.parse_args()

    run_dir     = Path(args.run_dir)
    out_dir     = Path(args.out_dir)
    scores_path = run_dir / 'cache' / 'neural_scores_test.parquet'
    db_path     = run_dir / 'cache' / 'data.duckdb'

    if not scores_path.exists():
        raise FileNotFoundError(f"Neural scores not found at {scores_path}. Run neural_score.py first.")

    con = duckdb.connect()
    con.execute(f"ATTACH '{db_path}' AS db")

    out_tsv = out_dir / 'matching_results_neural.tsv'

    # ── Build the country-aware threshold CASE expression ─────────
    if args.threshold is not None:
        # User explicitly overrode with a single global threshold
        threshold_expr = str(args.threshold)
        print(f"[INFO] Global threshold override: {args.threshold}")
    else:
        # Build CASE WHEN per country from the config dict above
        case_lines = "\n".join(
            f"            WHEN '{country}' THEN {thresh}"
            for country, thresh in COUNTRY_THRESHOLDS.items()
        )
        threshold_expr = f"""CASE s1.country
{case_lines}
            ELSE {DEFAULT_THRESHOLD}
        END"""
        print("[INFO] Using country-aware thresholds:")
        for country, thresh in COUNTRY_THRESHOLDS.items():
            print(f"         {country:<10}: {thresh}")
        print(f"         {'Default':<10}: {DEFAULT_THRESHOLD}")

    # ── Build the name similarity guard ───────────────────────────
    if NAME_SIM_GUARD > 0.0:
        name_sim_clause = f"""
                AND jaro_winkler_similarity(
                        lower(coalesce(s1.business_name, '')),
                        lower(coalesce(tgt.business_name, ''))
                    ) >= {NAME_SIM_GUARD}"""
        print(f"[INFO] Name similarity guard enabled: >= {NAME_SIM_GUARD}")
    else:
        name_sim_clause = ""
        print("[INFO] Name similarity guard disabled.")

    # ── Main query ────────────────────────────────────────────────
    query = f"""
    COPY (
        SELECT q.entity_id AS source1_entity_id,
               coalesce(x.ids, '') AS matched_entity_ids
        FROM db.test_s1 q
        LEFT JOIN (
            SELECT ns.qid,
                   string_agg(DISTINCT ns.tid, ',' ORDER BY ns.tid) AS ids
            FROM (
                -- Step 1: apply country-aware threshold + name similarity guard
                SELECT ns_raw.qid, ns_raw.tid, ns_raw.neural_score,
                       -- Step 2: Global Target Ownership — each tid goes to the highest-scoring qid only
                       row_number() OVER (
                           PARTITION BY ns_raw.tid
                           ORDER BY ns_raw.neural_score DESC
                       ) AS rn
                FROM read_parquet('{scores_path}') ns_raw
                JOIN db.test_s1 s1 ON s1.entity_id = ns_raw.qid
                JOIN (
                    SELECT entity_id, business_name FROM db.test_s2
                    UNION ALL
                    SELECT entity_id, business_name FROM db.test_s3
                ) tgt ON tgt.entity_id = ns_raw.tid
                WHERE ns_raw.neural_score >= {threshold_expr}{name_sim_clause}
            ) ns
            WHERE ns.rn = 1
            GROUP BY ns.qid
        ) x ON q.entity_id = x.qid
        ORDER BY q.rid
    ) TO '{out_tsv}' (FORMAT CSV, HEADER true, DELIMITER '\t', QUOTE '', NULL '')
    """

    print(f"\nGenerating {out_tsv} ...")
    con.execute(query)
    print(f"Done! Country-aware predictions written to: {out_tsv}")


if __name__ == '__main__':
    main()
