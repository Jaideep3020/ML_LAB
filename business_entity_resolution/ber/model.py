from __future__ import annotations

import json
import os
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .common import digest, fingerprint, read_json, sqlpath, write_json
from .features import NAMES, baseline_score, matrix
from .metrics import bootstrap_delta


def feature_glob(ctx,split,tag):
    return sqlpath(ctx.root/'cache'/'features'/split/tag/'part-*.parquet')


def training_arrays(ctx,tag,where="split='fit'",entity_limit=None,pair_limit=None,token='fit'):
    db=ctx.db()
    entity_limit=ctx.cfg['training_entity_limit'] if entity_limit is None else entity_limit
    pair_limit=pair_limit or ctx.cfg['max_training_pairs']
    limit=f'LIMIT {int(entity_limit)}' if entity_limit else ''
    db.execute(f"""CREATE OR REPLACE TEMP TABLE selected_queries AS
        SELECT DISTINCT qid FROM read_parquet('{feature_glob(ctx,'train',tag)}') WHERE {where}
        ORDER BY hash(qid,{ctx.cfg['seed']}) {limit}""")
    db.execute(f"""CREATE OR REPLACE TEMP VIEW pool AS
        SELECT f.*,row_number() OVER(PARTITION BY f.qid,label ORDER BY retrieval_score DESC,tid) AS negative_rank
        FROM read_parquet('{feature_glob(ctx,'train',tag)}') f JOIN selected_queries q USING(qid) WHERE {where}""")
    positives=db.execute('SELECT count(*) FROM pool WHERE label=1').fetchone()[0]
    if positives==0:
        raise ValueError('No retrieved positives for training')
    if positives>=pair_limit:
        raise MemoryError(f'{positives} positive pairs exceed pair budget {pair_limit}; raise budget after checking RAM, or train on entity shards')
    quota=pair_limit-positives
    db.execute(f"""CREATE OR REPLACE TEMP TABLE selected_pairs AS
        SELECT * FROM pool WHERE label=1 UNION ALL
        (SELECT * FROM pool WHERE label=0 AND negative_rank<={ctx.cfg['max_negatives_per_entity']}
         ORDER BY hash(qid,tid,{ctx.cfg['seed']}) LIMIT {quota})""")
    count=db.execute('SELECT count(*) FROM selected_pairs').fetchone()[0]
    if count==positives:
        raise ValueError('No negative pairs: broaden candidate retrieval')
    directory=ctx.root/'temp'/token
    directory.mkdir(parents=True,exist_ok=True)
    ctx.guard_disk(count*(len(NAMES)*4+1))
    x=np.memmap(directory/'x.bin',dtype=np.float32,mode='w+',shape=(count,len(NAMES)))
    y=np.memmap(directory/'y.bin',dtype=np.int8,mode='w+',shape=(count,))
    cur=db.execute('SELECT '+','.join(NAMES)+',label FROM selected_pairs ORDER BY qid,tid')
    offset=0
    for batch in cur.to_arrow_reader(30000):
        table=pa.Table.from_batches([batch]);size=len(table)
        x[offset:offset+size]=matrix(table);y[offset:offset+size]=table['label'].to_numpy();offset+=size
    x.flush();y.flush();db.close()
    x._ber_signature=fingerprint(dict(input=read_json(ctx.root/'reports'/'input.json'),tag=tag,where=where,
        entity_limit=entity_limit,pair_limit=pair_limit,config=ctx.cfg,features=NAMES))
    ctx.event('training_data',token=token,pairs=count,positives=positives,feature_bytes=count*len(NAMES)*4)
    return x,y


def params(ctx,trial=0):
    rng=np.random.default_rng(ctx.cfg['seed']+trial)
    return dict(objective='binary',metric='binary_logloss',verbosity=-1,
        num_threads=ctx.cfg['threads'],seed=ctx.cfg['seed'],deterministic=True,force_col_wise=True,
        histogram_pool_size=512,max_bin=127,
        learning_rate=.05 if trial==0 else float(rng.choice([.03,.05,.08])),
        num_leaves=31 if trial==0 else int(rng.choice([15,31,63,95])),
        min_data_in_leaf=50 if trial==0 else int(rng.choice([30,80,150,300])),
        feature_fraction=1. if trial==0 else float(rng.choice([.8,.9,1.])),
        lambda_l2=2. if trial==0 else float(rng.choice([1.,3.,8.,15.])))


def release_arrays(ctx,*arrays):
    for array in arrays:
        if isinstance(array,np.memmap):
            path=Path(array.filename).resolve()
            if not path.is_relative_to((ctx.root/'temp').resolve()):
                raise ValueError('Unexpected training-array location')
            array.flush();array._mmap.close();path.unlink(missing_ok=True)


def fit_lgb(ctx,x,y,parameters,destination):
    marker=destination.with_suffix('.json')
    signature=fingerprint(dict(data=getattr(x,'_ber_signature',None),parameters=parameters,rounds=ctx.cfg['boosting_rounds']))
    if destination.exists() and marker.exists() and read_json(marker).get('signature')==signature:
        return lgb.Booster(model_file=str(destination))
    model=lgb.train(parameters,lgb.Dataset(x,label=y,feature_name=NAMES,free_raw_data=True),
                    num_boost_round=ctx.cfg['boosting_rounds'])
    model.save_model(str(destination))
    write_json(marker,dict(signature=signature,features=NAMES))
    return model


def score_features(ctx,split,tag,models,weights,destination,baseline=False,where=None):
    destination=Path(destination)
    db=ctx.db()
    condition=f' WHERE {where}' if where else ''
    cursor=db.execute(f"SELECT * FROM read_parquet('{feature_glob(ctx,split,tag)}'){condition} ORDER BY qid,source,tid")
    schema=pa.schema([('qid',pa.int32()),('tid',pa.string()),('source',pa.int8()),('label',pa.int8()),
                      ('country',pa.string()),('score',pa.float32()),('fold',pa.int8()),('split',pa.string())])
    destination.parent.mkdir(parents=True,exist_ok=True)
    tmp=destination.with_suffix('.tmp')
    with pq.ParquetWriter(tmp,schema,compression='zstd') as writer:
        for batch in cursor.to_arrow_reader(50000):
            table=pa.Table.from_batches([batch])
            if baseline: scores=baseline_score(table)
            else:
                x=matrix(table);scores=np.zeros(len(table),dtype=np.float64)
                for model,weight in zip(models,weights):
                    pred=model.predict(x,num_threads=ctx.cfg['threads']) if isinstance(model,lgb.Booster) else model.predict_proba(x)[:,1]
                    scores+=weight*np.asarray(pred)
            cols=[table[n].combine_chunks() for n in ('qid','tid','source','label','country')]
            cols += [pa.array(scores,type=pa.float32()),table['fold'].combine_chunks(),table['split'].combine_chunks()]
            writer.write_table(pa.Table.from_arrays(cols,schema=schema))
    os.replace(tmp,destination);db.close()
    return destination


def evaluation_arrays(ctx,tag,score_path,where="s.split='tune'"):
    db=ctx.db()
    qids=[]
    for p in (ctx.root/'cache'/'candidates'/'train'/tag).glob('part-*.json'):
        qids.extend(read_json(p)['qids'])
    db.register('queried',pa.table({'qid':pa.array(qids,type=pa.int32())}))
    queries=db.execute(f"SELECT s.qid,s.matches,s.country FROM splits s JOIN queried q USING(qid) WHERE {where} ORDER BY s.qid").fetchall()
    if not queries:
        db.close();raise ValueError('No evaluation queries; retrieve a representative training sample first')
    qids=np.array([r[0] for r in queries],dtype=np.int32)
    truth=np.array([r[1] for r in queries],dtype=np.int32)
    countries=np.array([r[2] for r in queries])
    db.register('eval_ids',pa.table({'qid':qids}))
    table=db.execute(f"SELECT p.qid,p.label,p.source,p.score FROM read_parquet('{sqlpath(score_path)}') p JOIN eval_ids q USING(qid) ORDER BY p.qid").to_arrow_table()
    idx=np.searchsorted(qids,table['qid'].to_numpy())
    result=(qids,truth,countries,idx,table['label'].to_numpy(),table['source'].to_numpy(),table['score'].to_numpy())
    db.close();return result


def per_entity(arrays,t2,t3,gate=0):
    qids,truth,countries,idx,label,source,scores=arrays
    accept=scores>=np.where(source==2,t2,t3)
    if gate:
        maximum=np.zeros(len(qids),dtype=np.float32)
        np.maximum.at(maximum,idx,scores)
        accept &= maximum[idx]>=gate
    tp=np.bincount(idx,weights=accept*label,minlength=len(qids))
    predicted=np.bincount(idx,weights=accept,minlength=len(qids))
    fp=predicted-tp;fn=truth-tp
    denominator=1.25*tp+.25*fn+fp
    values=np.divide(1.25*tp,denominator,out=np.zeros_like(tp),where=denominator>0)
    values[(truth==0)&(predicted==0)]=1
    return values,predicted,tp


def choose_thresholds(arrays):
    grid=np.unique(np.r_[np.linspace(.05,.95,37),.97,.98,.99,.995,.999])
    best=(-1.,.5,.5,0.)
    for t in grid:
        score=float(per_entity(arrays,t,t)[0].mean())
        if score>best[0]: best=(score,float(t),float(t),0.)
    for _ in range(2):
        for src in (2,3):
            for t in grid:
                t2=float(t) if src==2 else best[1];t3=float(t) if src==3 else best[2]
                value=float(per_entity(arrays,t2,t3)[0].mean())
                if value>best[0]+1e-8:best=(value,t2,t3,0.)
    for gate in (0.,.5,.7,.85,.95,.99):
        value=float(per_entity(arrays,best[1],best[2],gate)[0].mean())
        if value>best[0]+1e-8: best=(value,best[1],best[2],gate)
    return dict(macro_f05=best[0],source2=best[1],source3=best[2],singleton_gate=best[3])


def summary(arrays,thresholds):
    values,predicted,tp=per_entity(arrays,thresholds['source2'],thresholds['source3'],thresholds.get('singleton_gate',0))
    qids,truth,countries,idx,label,source,scores=arrays
    single=truth==0
    report=dict(entities=len(qids),macro_f05=float(values.mean()),
        pair_precision=float(tp.sum()/max(1,predicted.sum())),pair_recall=float(tp.sum()/max(1,truth.sum())),
        candidate_recall=float(label.sum()/max(1,truth.sum())),
        singleton_accuracy=float(np.mean(predicted[single]==0)) if single.any() else None,
        singleton_false_positive_rate=float(np.mean(predicted[single]>0)) if single.any() else None,
        countries={c:float(values[countries==c].mean()) for c in np.unique(countries)},
        cardinality={str(n):float(values[truth==n].mean()) for n in np.unique(truth)})
    return report,values


def train(ctx,tag='main',final=False,search=False):
    selected_path=ctx.root/'models'/'selection.json'
    if final:
        selection=read_json(selected_path)
        x,y=training_arrays(ctx,tag,where="split IN ('fit','tune','holdout')",entity_limit=0,token='final')
        paths=[]
        for item in selection['models']:
            if item['type']=='lightgbm':
                path=ctx.root/'models'/f"final-{len(paths)}.txt"
                fit_lgb(ctx,x,y,item['params'],path)
            else:
                from catboost import CatBoostClassifier
                # Challenger has its own explicit training budget.
                take=min(len(y),ctx.cfg['catboost_pair_limit'])
                index=np.random.default_rng(ctx.cfg['seed']).choice(len(y),take,replace=False)
                model=CatBoostClassifier(**item['params']);model.fit(np.asarray(x[index]),np.asarray(y[index]))
                path=ctx.root/'models'/f'final-{len(paths)}.cbm';model.save_model(str(path))
            paths.append(dict(type=item['type'],path=path.name,weight=item['weight']))
        write_json(ctx.root/'models'/'selected.json',dict(models=paths,thresholds=selection['thresholds'],features=NAMES,final=True))
        release_arrays(ctx,x,y)
        ctx.event('train_final',models=len(paths));return
    x,y=training_arrays(ctx,tag,token='search')
    trials=[]
    for trial in range(ctx.cfg['search_trials'] if search else 1):
        path=ctx.root/'models'/f'trial-{trial}.txt'
        parameters=params(ctx,trial)
        model=fit_lgb(ctx,x,y,parameters,path)
        scored=score_features(ctx,'train',tag,[model],[1.],ctx.root/'cache'/f'trial-{trial}-scores.parquet',where="split='tune'")
        arrays=evaluation_arrays(ctx,tag,scored)
        thresholds=choose_thresholds(arrays)
        trials.append(dict(trial=trial,path=path.name,type='lightgbm',params=parameters,thresholds=thresholds))
        ctx.event('trial',trial=trial,**thresholds)
        del model,arrays
    trials.sort(key=lambda r:r['thresholds']['macro_f05'],reverse=True)
    release_arrays(ctx,x,y)
    write_json(ctx.root/'reports'/'search.json',trials)
    finalists=trials[:3] if search else trials[:1]
    # Three-fold OOF within development; holdout is never used here.
    oof_paths=[]
    for mi,item in enumerate(finalists):
        paths=[]
        for fold in range(3):
            xx,yy=training_arrays(ctx,tag,where=f"split<>'holdout' AND fold<>{fold}",entity_limit=0,token=f'oof-{mi}-{fold}')
            path=ctx.root/'models'/f'oof-{mi}-{fold}.txt'
            model=fit_lgb(ctx,xx,yy,item['params'],path)
            scores=score_features(ctx,'train',tag,[model],[1.],ctx.root/'cache'/f'oof-{mi}-{fold}.parquet',where=f"split<>'holdout' AND fold={fold}")
            paths.append(str(scores));release_arrays(ctx,xx,yy);del xx,yy,model
        target=ctx.root/'cache'/f'oof-{mi}.parquet'
        db=ctx.db();db.execute(f"COPY (SELECT * FROM read_parquet(?)) TO '{sqlpath(target)}' (FORMAT PARQUET)",[paths]);db.close()
        oof_paths.append(target)
    if search:
        from catboost import CatBoostClassifier
        catparams=dict(iterations=ctx.cfg['boosting_rounds'],depth=7,learning_rate=.05,loss_function='Logloss',
                       thread_count=ctx.cfg['threads'],random_seed=ctx.cfg['seed'],verbose=False,allow_writing_files=False)
        paths=[]
        for fold in range(3):
            xx,yy=training_arrays(ctx,tag,where=f"split<>'holdout' AND fold<>{fold}",pair_limit=ctx.cfg['catboost_pair_limit'],token=f'cat-{fold}')
            model=CatBoostClassifier(**catparams);model.fit(xx,yy)
            path=ctx.root/'models'/f'cat-{fold}.cbm';model.save_model(str(path))
            paths.append(str(score_features(ctx,'train',tag,[model],[1.],ctx.root/'cache'/f'cat-{fold}.parquet',where=f"split<>'holdout' AND fold={fold}")))
            release_arrays(ctx,xx,yy);del xx,yy,model
        target=ctx.root/'cache'/'oof-cat.parquet'
        db=ctx.db();db.execute(f"COPY (SELECT * FROM read_parquet(?)) TO '{sqlpath(target)}' (FORMAT PARQUET)",[paths]);db.close()
        finalists.append(dict(type='catboost',params=catparams));oof_paths.append(target)
    candidates=[]
    for i,path in enumerate(oof_paths):
        arrays=evaluation_arrays(ctx,tag,path,"s.split<>'holdout'")
        th=choose_thresholds(arrays);rep,values=summary(arrays,th)
        candidates.append(dict(indices=[i],weights=[1.],thresholds=th,report=rep,values=values,path=path))
    winner=max(candidates,key=lambda x:x['report']['macro_f05'])
    base=winner
    for j in range(len(finalists)):
        if j==base['indices'][0]:continue
        for weight in (.25,.5,.75):
            blend=ctx.root/'cache'/f'blend-{j}-{weight}.parquet'
            db=ctx.db()
            db.execute(f"""COPY (SELECT a.* REPLACE((a.score*{weight}+b.score*{1-weight})::FLOAT AS score)
                FROM read_parquet('{sqlpath(base['path'])}') a JOIN read_parquet('{sqlpath(oof_paths[j])}') b USING(qid,tid))
                TO '{sqlpath(blend)}' (FORMAT PARQUET)""")
            db.close()
            arrays=evaluation_arrays(ctx,tag,blend,"s.split<>'holdout'")
            th=choose_thresholds(arrays);rep,values=summary(arrays,th)
            if rep['macro_f05']>winner['report']['macro_f05']:
                interval=bootstrap_delta(winner['values'],values,repeats=200)
                if interval['low']>0 and all(rep['countries'][c]>=winner['report']['countries'][c]-.002 for c in rep['countries']):
                    winner=dict(indices=[base['indices'][0],j],weights=[weight,1-weight],thresholds=th,report=rep,values=values,path=blend)
    from .decisions import resolve_conflicts
    if read_json(ctx.root/'reports'/'audit.json')['shared_targets']==0:
        for margin in (0.,.01,.05,.1):
            adjusted=resolve_conflicts(ctx,winner['path'],margin)
            arrays=evaluation_arrays(ctx,tag,adjusted,"s.split<>'holdout'")
            rep,values=summary(arrays,winner['thresholds'])
            if rep['macro_f05']>winner['report']['macro_f05']:
                interval=bootstrap_delta(winner['values'],values,repeats=200)
                if interval['low']>0 and all(rep['countries'][c]>=winner['report']['countries'][c]-.002 for c in rep['countries']):
                    winner=dict(winner,report=rep,values=values,thresholds=dict(winner['thresholds'],conflict_margin=margin))
    selected=[];inference=[]
    # Fit development models for the one-time locked holdout evaluation.
    for i,weight in zip(winner['indices'],winner['weights']):
        item=finalists[i];item=dict(item,weight=weight);selected.append(item)
        if item['type']=='lightgbm':
            xx,yy=training_arrays(ctx,tag,where="split<>'holdout'",entity_limit=0,token=f'development-{i}')
            path=ctx.root/'models'/f'development-{i}.txt';fit_lgb(ctx,xx,yy,item['params'],path)
        else:
            from catboost import CatBoostClassifier
            xx,yy=training_arrays(ctx,tag,where="split<>'holdout'",pair_limit=ctx.cfg['catboost_pair_limit'],token=f'development-{i}')
            model=CatBoostClassifier(**item['params']);model.fit(xx,yy)
            path=ctx.root/'models'/f'development-{i}.cbm';model.save_model(str(path))
        inference.append(dict(type=item['type'],path=path.name,weight=weight));release_arrays(ctx,xx,yy);del xx,yy
    selection=dict(models=selected,thresholds=winner['thresholds'],oof=winner['report'],features=NAMES,config_hash=ctx.config_hash)
    write_json(selected_path,selection)
    write_json(ctx.root/'models'/'development.json',dict(models=inference,thresholds=winner['thresholds'],features=NAMES,final=False))
    ctx.event('selected',**winner['report'])


def load_models(ctx,name):
    manifest=read_json(ctx.root/'models'/f'{name}.json')
    if manifest['features']!=NAMES:raise ValueError('Model feature schema mismatch')
    models=[]
    for item in manifest['models']:
        path=ctx.root/'models'/item['path']
        if item['type']=='lightgbm':model=lgb.Booster(model_file=str(path))
        else:
            from catboost import CatBoostClassifier
            model=CatBoostClassifier();model.load_model(str(path))
        models.append(model)
    return manifest,models,[m['weight'] for m in manifest['models']]


def evaluate(ctx,tag='main',holdout=False):
    report_path=ctx.root/'reports'/('holdout.json' if holdout else 'evaluation.json')
    manifest,models,weights=load_models(ctx,'development')
    signature=fingerprint(manifest)
    if holdout and report_path.exists():
        previous=read_json(report_path)
        if previous['model_signature']!=signature:
            raise RuntimeError('Locked holdout already evaluated for a different selection')
        return previous
    if not holdout:
        report=read_json(ctx.root/'models'/'selection.json')['oof']
    else:
        path=score_features(ctx,'train',tag,models,weights,ctx.root/'cache'/'holdout-scores.parquet',where="split='holdout'")
        from .decisions import resolve_conflicts
        path=resolve_conflicts(ctx,path,manifest['thresholds'].get('conflict_margin'))
        arrays=evaluation_arrays(ctx,tag,path,"s.split='holdout'")
        report,_=summary(arrays,manifest['thresholds'])
    result=dict(report,model_signature=signature,thresholds=manifest['thresholds'])
    write_json(report_path,result);ctx.event('evaluate',**result);return result


def predict(ctx,split='test',tag='main',model='selected',baseline=False):
    candidates=read_json(ctx.root/'cache'/'candidates'/split/tag/'manifest.json')
    check=ctx.db()
    expected=check.execute(f'SELECT count(*) FROM {split}_s1').fetchone()[0]
    check.close()
    if not candidates.get('complete') or candidates['queries']!=expected:
        raise ValueError('Prediction requires candidates for every reference; a partial retrieval run cannot become a full submission')
    if baseline:manifest={'thresholds':{'source2':.9,'source3':.9,'singleton_gate':0}};models=[];weights=[]
    else:manifest,models,weights=load_models(ctx,model)
    scored=score_features(ctx,split,tag,models,weights,ctx.root/'cache'/f'{split}-scores.parquet',baseline=baseline)
    from .decisions import resolve_conflicts
    scored=resolve_conflicts(ctx,scored,manifest['thresholds'].get('conflict_margin'))
    th=manifest['thresholds'];db=ctx.db()
    db.execute(f"CREATE OR REPLACE TEMP VIEW scored AS SELECT *,max(score) OVER(PARTITION BY qid) AS maximum FROM read_parquet('{sqlpath(scored)}')")
    for filename,column,condition in (
        ('candidate_pairs.tsv','candidate_entity_ids','true'),
        ('matching_results.tsv','matched_entity_ids',f"score>=CASE WHEN source=2 THEN {th['source2']} ELSE {th['source3']} END AND maximum>={th.get('singleton_gate',0)}")):
        path=ctx.root/'output'/filename
        db.execute(f"""COPY (SELECT q.entity_id AS source1_entity_id,coalesce(x.ids,'') AS {column}
            FROM {split}_s1 q LEFT JOIN (SELECT qid,string_agg(DISTINCT tid,',' ORDER BY tid) AS ids FROM scored WHERE {condition} GROUP BY qid) x
            ON q.rid=x.qid ORDER BY q.rid) TO '{sqlpath(path)}' (FORMAT CSV,HEADER true,DELIMITER '\t',QUOTE '',NULL '')""")
    db.close()
    write_json(ctx.root/'reports'/'prediction.json',dict(dataset=split,tag=tag,baseline=baseline,thresholds=th,
        candidate_manifest=candidates))
    ctx.event('predict',dataset=split,baseline=baseline)
