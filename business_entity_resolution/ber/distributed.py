"""Resumable, bounded-storage test inference over explicit, disjoint RID ranges."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import os
import shutil
import time
import pyarrow as pa
import pyarrow.parquet as pq
from .common import Context, digest, fingerprint, read_json, sqlpath, write_json
from .retrieval import CANDIDATE_SCHEMA, _start_retriever, _retrieve_chunk
from .features import features
from .model import load_models, score_features


def initialize(ctx):
    db = ctx.db()
    for source in (1, 2, 3):
        path = ctx.root / 'cache' / 'parquet' / f'test_s{source}' / '**/*.parquet'
        db.execute(f"CREATE TABLE IF NOT EXISTS test_s{source} AS SELECT * FROM read_parquet('{sqlpath(path)}',hive_partitioning=true)")
    db.execute('CHECKPOINT')
    db.close()


def run(ctx, start, end, reuse=None):
    manifest, models, weights = load_models(ctx, 'selected')
    if manifest['thresholds'].get('conflict_margin') is not None:
        raise ValueError('Global conflict resolution cannot be applied independently per shard')
    cfg = {k:v for k,v in ctx.cfg.items() if k not in ('threads','memory_limit','retrieval_workers','feature_workers','disk_reserve_gb')}
    signature = fingerprint(dict(model=manifest, config=cfg,
        weights=[digest(ctx.root/'models'/m['path']) for m in manifest['models']],
        aliases=digest(ctx.root/'cache/aliases-v2.json'), input=read_json(ctx.root/'reports/input.json')['sha256']))
    root = ctx.root / 'output' / 'distributed'
    root.mkdir(parents=True, exist_ok=True)
    db = ctx.db()
    rows = db.execute('SELECT rid,business_name,business_address,country FROM test_s1 WHERE rid>=? AND rid<=? ORDER BY rid',[start,end]).fetchall()
    db.close()
    if len(rows) != end-start+1:
        raise ValueError('Missing or non-contiguous query IDs in assigned range')
    started = time.time()
    with ProcessPoolExecutor(max_workers=ctx.cfg['retrieval_workers'], initializer=_start_retriever,
                            initargs=(str(ctx.root),ctx.cfg,'test')) as pool:
        for offset in range(0,len(rows),1000):
            group = rows[offset:offset+1000]
            lo, hi = group[0][0], group[-1][0]
            tag = f'dist-{lo:07d}-{hi:07d}'
            done = root / (tag+'.json')
            outputs = [root/(tag+'-matching.tsv'), root/(tag+'-candidates.tsv')]
            if done.exists():
                meta=read_json(done)
                if meta['signature']!=signature or any(not p.exists() or digest(p)!=meta['hashes'][p.name] for p in outputs):
                    raise ValueError('Checkpoint changed: '+str(done))
                continue
            ctx.guard_disk(2 << 30)
            cand = ctx.root/'cache/candidates/test'/tag
            cand.mkdir(parents=True,exist_ok=True)
            path = cand/'part-000000.parquet'
            qids = [r[0] for r in group]
            old = Path(reuse)/f'part-{(lo-1)//1000:06d}.parquet' if reuse and (lo-1)%1000==0 else None
            if old and old.with_suffix('.json').exists() and read_json(old.with_suffix('.json'))['qids']==qids:
                shutil.copy2(old,path)
                pairs=pq.ParquetFile(path).metadata.num_rows
            else:
                output=[]
                for chunk in pool.map(_retrieve_chunk,[(group[i:i+32],200) for i in range(0,len(group),32)]):
                    output.extend(chunk)
                columns=list(zip(*output)) if output else [[] for _ in CANDIDATE_SCHEMA]
                table=pa.Table.from_arrays([pa.array(v,type=f.type) for f,v in zip(CANDIDATE_SCHEMA,columns)],schema=CANDIDATE_SCHEMA)
                pq.write_table(table,path,compression='zstd')
                pairs=len(output)
                del output, columns, table
            write_json(path.with_suffix('.json'),dict(signature=signature,qids=qids,pairs=pairs,queries=len(group)))
            write_json(cand/'manifest.json',dict(complete=True,queries=len(group),pairs=pairs))
            if pairs:
                features(ctx,'test',tag)
                score=ctx.root/'cache'/(tag+'-scores.parquet')
                score_features(ctx,'test',tag,models,weights,score)
            db=ctx.db()
            if pairs:
                db.execute(f"CREATE TEMP VIEW scored AS SELECT *,max(score) OVER(PARTITION BY qid) AS maximum FROM read_parquet('{sqlpath(score)}')")
            else:
                db.execute('CREATE TEMP TABLE scored(qid INTEGER,tid VARCHAR,source INTEGER,score FLOAT,maximum FLOAT)')
            th=manifest['thresholds']
            conditions=[f"score>=CASE WHEN source=2 THEN {th['source2']} ELSE {th['source3']} END AND maximum>={th.get('singleton_gate',0)}",'true']
            for dest,col,condition in zip(outputs,['matched_entity_ids','candidate_entity_ids'],conditions):
                tmp=dest.with_suffix('.tmp')
                db.execute(f"""COPY (SELECT q.entity_id AS source1_entity_id,coalesce(x.ids,'') AS {col}
                    FROM test_s1 q LEFT JOIN (SELECT qid,string_agg(DISTINCT tid,',' ORDER BY tid) ids FROM scored WHERE {condition} GROUP BY qid) x
                    ON q.rid=x.qid WHERE q.rid BETWEEN {lo} AND {hi} ORDER BY q.rid)
                    TO '{sqlpath(tmp)}' (FORMAT CSV,HEADER true,DELIMITER '\t',QUOTE '',NULL '')""")
                os.replace(tmp,dest)
            db.close()
            write_json(done,dict(signature=signature,start=lo,end=hi,queries=len(group),pairs=pairs,
                                hashes={p.name:digest(p) for p in outputs}))
            # Only remove this runner's exact temporary shard directories after commit.
            for folder in (cand,ctx.root/'cache/features/test'/tag):
                if folder.exists() and folder.name==tag and folder.resolve().is_relative_to(ctx.root/'cache'):
                    shutil.rmtree(folder)
            if pairs:
                score.unlink(missing_ok=True)
            ctx.event('distributed_batch',start=lo,end=hi,completed=offset+len(group),assigned=len(rows),seconds=time.time()-started)
    write_json(root/f'assignment-{start}-{end}.done.json',dict(start=start,end=end,signature=signature,complete=True))


def merge(ctx, folders):
    """Require complete, non-overlapping coverage and verify each file before merge."""
    checkpoints={}
    for folder in folders:
        for p in Path(folder).glob('dist-*.json'):
            m=read_json(p)
            # Ignore the explicitly separate ten-query portability probe.
            if m['start']==1 and m['end']==10: continue
            if m['start'] in checkpoints: raise ValueError('Duplicate shard: '+str(p))
            checkpoints[m['start']]=(p,m)
    db=ctx.db()
    ids=db.execute('SELECT rid,entity_id FROM test_s1 ORDER BY rid').fetchall()
    db.close()
    position=0;signature=None
    names=['matching_results.tsv','candidate_pairs.tsv']
    tmps=[ctx.root/'output'/(n+'.merge.tmp') for n in names]
    import contextlib
    with contextlib.ExitStack() as stack:
        writers=[stack.enter_context(open(p,'w',encoding='utf-8',newline='')) for p in tmps]
        writers[0].write('source1_entity_id\tmatched_entity_ids\n')
        writers[1].write('source1_entity_id\tcandidate_entity_ids\n')
        for lo,(marker,m) in sorted(checkpoints.items()):
            if position>=len(ids) or lo!=ids[position][0]: raise ValueError('Missing or overlapping shard at '+str(lo))
            if signature is None: signature=m['signature']
            if signature!=m['signature']: raise ValueError('Model/configuration differs between machines')
            tag=marker.stem
            paths=[marker.parent/(tag+'-matching.tsv'),marker.parent/(tag+'-candidates.tsv')]
            for p in paths:
                if digest(p)!=m['hashes'][p.name]: raise ValueError('Corrupt transfer: '+str(p))
            with open(paths[0],encoding='utf-8') as a,open(paths[1],encoding='utf-8') as b:
                next(a);next(b)
                count=0
                import itertools
                for x,y in itertools.zip_longest(a,b):
                    if x is None or y is None: raise ValueError('Unequal shard row counts')
                    if position>=len(ids): raise ValueError('Too many rows')
                    expected=ids[position][1]
                    if x.split('\t',1)[0]!=expected or y.split('\t',1)[0]!=expected: raise ValueError('Query identity/order mismatch')
                    writers[0].write(x);writers[1].write(y)
                    position+=1;count+=1
                if count!=m['queries'] or ids[position-1][0]!=m['end']: raise ValueError('Invalid shard bounds')
        if position!=len(ids): raise ValueError(f'Incomplete coverage: {position}/{len(ids)}')
    for tmp,name in zip(tmps,names):
        dest=ctx.root/'output'/name
        if dest.exists() and not dest.with_suffix('.pre-distributed.tsv').exists():
            shutil.copy2(dest,dest.with_suffix('.pre-distributed.tsv'))
        os.replace(tmp,dest)
    from .delivery import validate
    validate(ctx)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',required=True);p.add_argument('--config',required=True)
    p.add_argument('--start',type=int);p.add_argument('--end',type=int)
    p.add_argument('--initialize',action='store_true');p.add_argument('--reuse')
    p.add_argument('--merge',nargs='+')
    args=p.parse_args();ctx=Context(args.run_dir,args.config)
    if args.initialize: initialize(ctx)
    if args.merge: merge(ctx,args.merge)
    if args.start is not None and args.end is not None: run(ctx,args.start,args.end,args.reuse)

if __name__=='__main__': main()
