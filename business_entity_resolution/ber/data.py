from __future__ import annotations

import csv
import io
import os
from pathlib import Path
import shutil
import zipfile

import numpy as np
import pyarrow as pa

from .common import batches, digest, read_json, sqlpath, write_json
from .text import normalize


def prepare(ctx, archive):
    archive = Path(archive)
    manifest_path = ctx.root / "reports" / "input.json"
    sha = digest(archive)
    if manifest_path.exists() and read_json(manifest_path)["sha256"] != sha:
        raise ValueError("Archive changed. Use a new run directory.")
    ctx.guard_disk(10 << 30)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            parts = Path(member.filename).parts
            if not parts or parts[0] != "student_resource" or member.is_dir() or parts[-1].startswith("."):
                continue
            if ".." in parts:
                raise ValueError("Unsafe archive member")
            destination = ctx.root / "data" / Path(*parts[1:])
            if destination.exists() and destination.stat().st_size == member.file_size:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            tmp = destination.with_suffix(destination.suffix + ".tmp")
            with z.open(member) as src, open(tmp, "wb") as out:
                shutil.copyfileobj(src, out, 8 << 20)
            os.replace(tmp, destination)
    write_json(manifest_path, {"archive": str(archive.resolve()), "sha256": sha})
    db = ctx.db()
    counts = {}
    for split in ("train", "test"):
        for source in (1, 2, 3):
            table = f"{split}_s{source}"
            path = ctx.root / "data" / "dataset" / split / f"{split}_source{source}.tsv"
            exists = db.execute("SELECT count(*) FROM information_schema.tables WHERE table_name=?", [table]).fetchone()[0]
            if not exists:
                mapping = Path(__file__).resolve().parents[2] / 'artifacts' / 'row_ids' / f'{table}.parquet'
                rid_expression = "row_number() OVER (ORDER BY entity_id)::INTEGER"
                mapping_join = ''
                if mapping.exists() and read_json(mapping.parent/'manifest.json')['archive_sha256'] == sha:
                    rid_expression = 'm.rid'
                    mapping_join = f"JOIN read_parquet('{sqlpath(mapping)}') m USING(entity_id)"
                db.execute(f"""CREATE TABLE {table} AS
                    SELECT {rid_expression} AS rid,
                        entity_id, coalesce(business_name,'') AS business_name,
                        coalesce(business_address,'') AS business_address,
                        coalesce(country,'') AS country
                    FROM read_csv('{sqlpath(path)}', delim='\t', header=true, all_varchar=true,
                                  quote='', null_padding=false, strict_mode=true) {mapping_join}""")
            counts[table] = db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            dest = ctx.root / "cache" / "parquet" / table
            done = dest / "complete.json"
            if not done.exists():
                dest.mkdir(parents=True, exist_ok=True)
                db.execute(f"COPY {table} TO '{sqlpath(dest)}' (FORMAT PARQUET, PARTITION_BY(country), OVERWRITE_OR_IGNORE true)")
                write_json(done, {"rows": counts[table], "archive": sha})
            ctx.event("prepare", table=table, rows=counts[table])
    gt = ctx.root / "data" / "dataset" / "train" / "train_ground_truth.tsv"
    db.execute(f"""CREATE TABLE IF NOT EXISTS truth_raw AS SELECT source1_entity_id,
        coalesce(matched_entity_ids,'') AS matched_entity_ids
        FROM read_csv('{sqlpath(gt)}',delim='\t',header=true,all_varchar=true,quote='')""")
    db.execute("""CREATE TABLE IF NOT EXISTS truth AS
        SELECT q.rid AS qid, t.source1_entity_id, trim(mid) AS tid
        FROM truth_raw t JOIN train_s1 q ON q.entity_id=t.source1_entity_id,
        unnest(string_split(t.matched_entity_ids,',')) AS x(mid) WHERE trim(mid)<>''""")
    db.execute("CHECKPOINT")
    db.close()
    write_json(ctx.root / "reports" / "counts.json", counts)


def audit(ctx):
    db = ctx.db()
    result = {"sources": {}, "fatal": []}
    for split in ("train", "test"):
        for source in (1, 2, 3):
            table = f"{split}_s{source}"
            row = db.execute(f"""SELECT count(*), count(*)-count(DISTINCT entity_id),
                count(*) FILTER(WHERE NOT starts_with(entity_id,'S{source}-')),
                count(*) FILTER(WHERE business_name=''),count(*) FILTER(WHERE business_address='') FROM {table}""").fetchone()
            result["sources"][table] = dict(zip(["rows", "duplicate_ids", "bad_prefix", "missing_name", "missing_address"], row))
            result["sources"][table]["countries"] = dict(db.execute(f"SELECT country,count(*) FROM {table} GROUP BY country").fetchall())
            if row[1] or row[2]:
                result["fatal"].append(table + ": invalid IDs")
    checks = {
        "missing_truth_rows": "SELECT count(*) FROM train_s1 q ANTI JOIN truth_raw t ON q.entity_id=t.source1_entity_id",
        "unknown_truth_references": "SELECT count(*) FROM truth_raw t ANTI JOIN train_s1 q ON q.entity_id=t.source1_entity_id",
        "duplicate_truth_rows": "SELECT count(*)-count(DISTINCT source1_entity_id) FROM truth_raw",
        "duplicate_truth_pairs": "SELECT count(*) FROM (SELECT qid,tid FROM truth GROUP BY ALL HAVING count(*)>1)",
        "invalid_targets": "SELECT count(*) FROM truth ANTI JOIN (SELECT entity_id FROM train_s2 UNION ALL SELECT entity_id FROM train_s3) t ON tid=t.entity_id",
        "shared_targets": "SELECT count(*) FROM (SELECT tid FROM truth GROUP BY tid HAVING count(DISTINCT qid)>1)",
        "cross_country_pairs": "SELECT count(*) FROM truth t JOIN train_s1 q ON t.qid=q.rid JOIN (SELECT entity_id,country FROM train_s2 UNION ALL SELECT entity_id,country FROM train_s3) x ON t.tid=x.entity_id WHERE q.country<>x.country",
    }
    for key, sql in checks.items():
        result[key] = db.execute(sql).fetchone()[0]
        if key not in ("shared_targets", "cross_country_pairs") and result[key]:
            result["fatal"].append(key)
    result["cardinality"] = dict(db.execute("SELECT n,count(*) FROM (SELECT q.rid,count(t.tid) AS n FROM train_s1 q LEFT JOIN truth t ON q.rid=t.qid GROUP BY q.rid) GROUP BY n").fetchall())
    result["country_blocking_safe_on_training"] = result["cross_country_pairs"] == 0
    write_json(ctx.root / "reports" / "audit.json", result)
    db.close()
    if result["fatal"]:
        raise ValueError(f"Data audit failed: {result['fatal']}")
    ctx.event("audit", **{k: v for k, v in result.items() if k not in ("sources", "cardinality")})
    return result


def make_splits(ctx):
    db = ctx.db()
    if db.execute("SELECT count(*) FROM information_schema.tables WHERE table_name='splits'").fetchone()[0]:
        db.close()
        return
    n = db.execute("SELECT max(rid) FROM train_s1").fetchone()[0]
    parent = np.arange(n + 1, dtype=np.int32)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    # Keep Python UDF execution single-threaded. Subsequent native aggregation
    # restores the configured thread count.
    db.execute('SET threads=1')
    db.create_function("norm", normalize, ["VARCHAR"], "VARCHAR")
    # All signatures in each labeled business participate, not only S1 signatures.
    db.execute("""CREATE TABLE IF NOT EXISTS split_signatures AS
        SELECT rid AS qid, country, norm(business_name) AS n, norm(business_address) AS a FROM train_s1
        UNION ALL SELECT t.qid,r.country,norm(r.business_name),norm(r.business_address)
        FROM truth t JOIN (SELECT * FROM train_s2 UNION ALL SELECT * FROM train_s3) r ON t.tid=r.entity_id""")
    db.remove_function('norm')
    db.execute('CHECKPOINT')
    ctx.event('split_signatures_ready')
    db.execute('SET threads=?',[ctx.cfg['threads']])
    # Emit only edges that actually connect different reference businesses.
    # Windowing all repeated aliases would materialize millions of no-op edges.
    edges = db.execute("""SELECT s.qid,g.leader FROM split_signatures s JOIN
        (SELECT country,n,a,min(qid) AS leader FROM split_signatures WHERE n<>'' AND a<>''
         GROUP BY country,n,a HAVING count(DISTINCT qid)>1) g USING(country,n,a)
        WHERE s.qid<>g.leader UNION ALL
        SELECT t.qid,g.leader FROM truth t JOIN
        (SELECT tid,min(qid) AS leader FROM truth GROUP BY tid HAVING count(DISTINCT qid)>1) g USING(tid)
        WHERE t.qid<>g.leader""")
    for rows in batches(edges, 100000):
        for a, b in rows:
            union(a, b)
    roots = np.array([find(i) for i in range(1, n+1)], dtype=np.int32)
    db.register("groups_input", pa.table({"qid": np.arange(1,n+1,dtype=np.int32), "group_id": roots}))
    db.execute(f"""CREATE TABLE splits AS WITH base AS (
        SELECT g.*, q.country,count(t.tid)::INTEGER AS matches,
          hash(g.group_id,{ctx.cfg['seed']}) % 100 AS bucket,
          hash(g.group_id,{ctx.cfg['seed'] + 1}) % 3 AS fold
        FROM groups_input g JOIN train_s1 q ON g.qid=q.rid LEFT JOIN truth t ON t.qid=g.qid
        GROUP BY g.qid,g.group_id,q.country)
        SELECT *,CASE WHEN bucket<80 THEN 'fit' WHEN bucket<90 THEN 'tune' ELSE 'holdout' END AS split FROM base""")
    stats = db.execute("SELECT split,country,least(matches,8),count(*) FROM splits GROUP BY ALL ORDER BY ALL").fetchall()
    write_json(ctx.root / "reports" / "splits.json", {"seed": ctx.cfg["seed"], "strata": stats,
        "method": "Connected business groups, deterministic hash allocation; approximate stratification audited by country/cardinality"})
    db.execute("CHECKPOINT")
    db.close()
    ctx.event("split", entities=n, groups=len(np.unique(roots)))
