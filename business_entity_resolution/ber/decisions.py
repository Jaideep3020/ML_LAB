from __future__ import annotations

from pathlib import Path

from .common import sqlpath


def resolve_conflicts(ctx,path,margin):
    """At most one owner per target, never at most one target per reference.

    Call only after training-label exclusivity has been audited and this optional
    policy has improved independent development predictions.
    """
    if margin is None:return Path(path)
    source=Path(path)
    out=source.with_name(source.stem+f'-conflicts-{margin}.parquet')
    db=ctx.db()
    db.execute(f"""COPY (WITH ranked AS (
        SELECT *,row_number() OVER(PARTITION BY tid ORDER BY score DESC,qid) AS owner_rank,
        first_value(score) OVER(PARTITION BY tid ORDER BY score DESC,qid) AS best,
        nth_value(score,2) OVER(PARTITION BY tid ORDER BY score DESC,qid
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS second
        FROM read_parquet('{sqlpath(source)}'))
        SELECT * EXCLUDE(owner_rank,best,second) REPLACE(
          CASE WHEN owner_rank=1 AND best-coalesce(second,0)>={float(margin)} THEN score ELSE 0 END AS score)
        FROM ranked) TO '{sqlpath(out)}' (FORMAT PARQUET)""")
    db.close();return out
