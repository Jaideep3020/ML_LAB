from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from .common import read_json,sqlpath,write_json
from .model import (training_arrays,fit_lgb,score_features,evaluation_arrays,
                    choose_thresholds,summary,load_models,release_arrays)


def country_transfer(ctx,tag='main'):
    selection=read_json(ctx.root/'models'/'selection.json')
    reports=[]
    for origin,target in [('US','India'),('India','US')]:
        models=[];weights=[]
        for i,item in enumerate(selection['models']):
            xx,yy=training_arrays(ctx,tag,where=f"split='fit' AND country='{origin}'",token=f'transfer-{origin}-{i}',
                                   pair_limit=ctx.cfg['catboost_pair_limit'] if item['type']=='catboost' else None)
            if item['type']=='lightgbm':
                model=fit_lgb(ctx,xx,yy,item['params'],ctx.root/'models'/f'transfer-{origin}-{i}.txt')
            else:
                from catboost import CatBoostClassifier
                model=CatBoostClassifier(**item['params']);model.fit(xx,yy)
            models.append(model);weights.append(item['weight']);release_arrays(ctx,xx,yy);del xx,yy
        scored=score_features(ctx,'train',tag,models,weights,ctx.root/'cache'/f'transfer-{origin}.parquet',where="split='tune'")
        # Calibrate using the origin country's labels only, then freeze thresholds.
        arrays=evaluation_arrays(ctx,tag,scored,f"s.split='tune' AND s.country='{origin}'")
        thresholds=choose_thresholds(arrays)
        arrays=evaluation_arrays(ctx,tag,scored,f"s.split='tune' AND s.country='{target}'")
        result,_=summary(arrays,thresholds)
        reports.append(dict(train_country=origin,evaluation_country=target,thresholds=thresholds,**result))
        ctx.event('country_transfer',**reports[-1])
    write_json(ctx.root/'reports'/'country-transfer.json',reports)
    return reports


def mine_negatives(ctx,score_path,per_entity=20):
    db=ctx.db()
    # Require out-of-fold data provenance: this utility only accepts generated OOF artifacts.
    path=Path(score_path).resolve()
    if path.parent!= (ctx.root/'cache').resolve() or not path.name.startswith('oof-'):
        raise ValueError('Use a generated out-of-fold score artifact')
    out=ctx.root/'cache'/'mined-negatives.parquet'
    db.execute(f"""COPY (SELECT qid,tid,score FROM read_parquet('{sqlpath(path)}')
        WHERE label=0 AND split<>'holdout'
        QUALIFY row_number() OVER(PARTITION BY qid ORDER BY score DESC,tid)<={int(per_entity)})
        TO '{sqlpath(out)}' (FORMAT PARQUET)""")
    rows=db.execute(f"SELECT count(*) FROM read_parquet('{sqlpath(out)}')").fetchone()[0]
    db.close()
    write_json(ctx.root/'reports'/'negative-mining.json',dict(rows=rows,source=str(path),
        note='Additional hard negatives for a subsequent independent grouped validation experiment; not silently added to the selected model.'))
    return out


def error_analysis(ctx,tag='main'):
    manifest,models,weights=load_models(ctx,'development')
    scored=score_features(ctx,'train',tag,models,weights,ctx.root/'cache'/'error-scores.parquet',where="split='tune'")
    db=ctx.db();thresholds=manifest['thresholds']
    dest=ctx.root/'reports'/'error_examples.tsv'
    db.execute(f"""COPY (
        SELECT p.qid,p.tid,p.label,p.score,q.country,q.business_name AS reference_name,
               q.business_address AS reference_address,r.business_name AS target_name,r.business_address AS target_address,
               CASE WHEN p.label=0 THEN 'false_positive' ELSE 'false_negative' END AS error
        FROM read_parquet('{sqlpath(scored)}') p JOIN train_s1 q ON q.rid=p.qid
        JOIN (SELECT entity_id,business_name,business_address FROM train_s2 UNION ALL SELECT entity_id,business_name,business_address FROM train_s3) r ON p.tid=r.entity_id
        WHERE (p.score>=CASE WHEN p.source=2 THEN {thresholds['source2']} ELSE {thresholds['source3']} END)<>(p.label=1)
        ORDER BY abs(p.score-.5) DESC LIMIT 500
        ) TO '{sqlpath(dest)}' (FORMAT CSV,HEADER true,DELIMITER '\t')""")
    # Retrieval misses are reported separately and include absent candidate positives.
    misses=ctx.root/'reports'/'retrieval_misses.tsv'
    db.execute(f"""COPY (SELECT t.source1_entity_id,t.tid,q.country,q.business_name,q.business_address
        FROM truth t JOIN train_s1 q ON q.rid=t.qid JOIN splits s ON s.qid=t.qid
        ANTI JOIN read_parquet('{sqlpath(scored)}') p ON p.qid=t.qid AND p.tid=t.tid
        WHERE s.split='tune' AND t.qid IN (SELECT DISTINCT qid FROM read_parquet('{sqlpath(scored)}'))
        LIMIT 500) TO '{sqlpath(misses)}' (FORMAT CSV,HEADER true,DELIMITER '\t')""")
    db.close()
    return dest
