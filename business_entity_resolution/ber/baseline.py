"""Fast, conservative full-corpus baseline independent of the expensive search."""
from __future__ import annotations

from .common import sqlpath,write_json


def exact_baseline(ctx,dataset='test'):
    db=ctx.db()
    # This baseline uses native Unicode NFC and regular expressions so it can be
    # run before the richer Python/FTS stages. Ambiguous abbreviations are retained.
    def norm(col):
        return f"trim(regexp_replace(lower(nfc_normalize({col})), '[^\\p{{L}}\\p{{M}}\\p{{N}}]+', ' ', 'g'))"
    for source in (1,2,3):
        db.execute(f"""CREATE TABLE IF NOT EXISTS {dataset}_baseline_s{source} AS
            SELECT rid,entity_id,country,
            regexp_replace({norm('business_name')},' (inc|incorporated|llc|ltd|limited|corp|corporation|pvt|private|sarl|sas)( (inc|incorporated|llc|ltd|limited|corp|corporation|pvt|private|sarl|sas))*$','') AS n,
            {norm('business_address')} AS a FROM {dataset}_s{source}""")
        ctx.event('baseline_normalized',dataset=dataset,source=source)
    db.execute(f"""CREATE OR REPLACE TEMP TABLE baseline_pairs AS
        SELECT q.rid AS qid,t.entity_id AS tid FROM {dataset}_baseline_s1 q
        JOIN (SELECT * FROM {dataset}_baseline_s2 UNION ALL SELECT * FROM {dataset}_baseline_s3) t
        ON q.country=t.country AND q.n=t.n AND q.a=t.a WHERE q.n<>'' AND q.a<>''""")
    pairs=db.execute('SELECT count(*) FROM baseline_pairs').fetchone()[0]
    if dataset=='train':
        row=db.execute("""WITH p AS(SELECT qid,count(*) AS predicted FROM baseline_pairs GROUP BY qid),
            tp AS(SELECT b.qid,count(*) AS tp FROM baseline_pairs b JOIN truth t USING(qid,tid) GROUP BY b.qid),
            actual AS(SELECT qid,count(*) AS n FROM truth GROUP BY qid)
            SELECT avg(CASE WHEN coalesce(a.n,0)=0 THEN CASE WHEN coalesce(p.predicted,0)=0 THEN 1. ELSE 0. END
                ELSE 1.25*coalesce(tp.tp,0)/(.25*a.n+coalesce(p.predicted,0)) END)
            FROM train_s1 q LEFT JOIN p ON q.rid=p.qid LEFT JOIN tp ON q.rid=tp.qid LEFT JOIN actual a ON q.rid=a.qid""").fetchone()
        report=dict(dataset=dataset,pairs=pairs,macro_f05=row[0],method='exact normalized name/address agreement; no learned parameters')
        write_json(ctx.root/'reports'/'baseline-train.json',report)
    else:
        for filename,column in [('matching_results.tsv','matched_entity_ids'),('candidate_pairs.tsv','candidate_entity_ids')]:
            path=ctx.root/'output'/filename
            db.execute(f"""COPY (SELECT q.entity_id AS source1_entity_id,coalesce(p.ids,'') AS {column}
                FROM test_s1 q LEFT JOIN (SELECT qid,string_agg(DISTINCT tid,',' ORDER BY tid) AS ids FROM baseline_pairs GROUP BY qid) p
                ON p.qid=q.rid ORDER BY q.rid) TO '{sqlpath(path)}' (FORMAT CSV,HEADER true,DELIMITER '\t',QUOTE '',NULL '')""")
        report=dict(dataset=dataset,pairs=pairs,baseline=True,method='exact normalized name/address agreement',
                    note='Conservative baseline, not the trained matcher. Every candidate is scored by the deterministic exact-agreement rule.')
        write_json(ctx.root/'reports'/'prediction.json',report)
    db.close();ctx.event('baseline',**report)
    return report
