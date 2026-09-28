from __future__ import annotations

from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor
import math
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

from .common import batches, fingerprint, read_json, sqlpath, write_json
from .text import core_name, fold, grams, numeric, postal, views


NAMES = [
    'name_exact','name_core_exact','name_fold_exact','name_ratio','name_jaro','name_token_sort',
    'name_token_set','name_jaccard','name_containment','name_gram_jaccard','name_acronym',
    'name_length_ratio','name_missing','address_exact','address_ratio','address_jaro',
    'address_token_sort','address_token_set','address_jaccard','address_containment',
    'address_gram_jaccard','address_length_ratio','address_missing','number_jaccard',
    'number_conflict','number_shared','postal_agree','postal_conflict','script_equal',
    'country_equal','rank_inverse','retrieval_score','pass_count','exact_pass',
    'name_pass','address_pass','name_gram_pass','address_gram_pass','combined_pass',
    'candidate_count_log','name_address_product','name_only_evidence','address_only_evidence',
    'name_tokens_log','address_tokens_log',
]


def jac(a,b):
    shared = len(a & b)
    total = len(a) + len(b) - shared
    return shared / total if total else 0.


def containment(a,b):
    return len(a&b)/min(len(a),len(b)) if a and b else 0.


def ratio_len(a,b):
    return min(len(a),len(b))/max(len(a),len(b)) if a and b else 0.


@lru_cache(maxsize=3000)
def cached(name,addr):
    n,a,core,nfold,script=views(name,addr)
    nw, aw = n.split(), a.split()
    return (n,a,core,nfold,script,set(nw),set(aw),grams(n),grams(a),numeric(a),postal(a),
            ' '.join(sorted(nw)), ' '.join(sorted(aw)), ''.join(t[0] for t in nw))


def pair_features(qname,qaddr,qcountry,tname,taddr,tcountry,rank,score,passes,count):
    n,a,core,nf,sc,nt,at,ng,ag,num,pc,ns,ads,ac=cached(qname,qaddr)
    m,b,mcore,mf,tc,mt,bt,mg,bg,nums,pcs,ms,bds,bc=cached(tname,taddr)
    ni, ai, nui, pci = len(nt & mt), len(at & bt), len(num & nums), len(pc & pcs)
    nr=fuzz.ratio(n,m)/100 if n and m else 0.
    ar=fuzz.ratio(a,b)/100 if a and b else 0.
    values=[
        bool(n and n==m),bool(core and core==mcore),bool(nf and nf==mf),nr,
        JaroWinkler.normalized_similarity(n,m) if n and m else 0.,
        fuzz.ratio(ns,ms)/100 if n and m else 0.,fuzz.token_set_ratio(n,m)/100 if n and m else 0.,
        ni/(len(nt)+len(mt)-ni) if nt or mt else 0.,ni/min(len(nt),len(mt)) if nt and mt else 0.,jac(ng,mg),
        bool(nt and mt and ac==bc),
        ratio_len(n,m),not(n and m),bool(a and a==b),ar,
        JaroWinkler.normalized_similarity(a,b) if a and b else 0.,
        fuzz.ratio(ads,bds)/100 if a and b else 0.,fuzz.token_set_ratio(a,b)/100 if a and b else 0.,
        ai/(len(at)+len(bt)-ai) if at or bt else 0.,ai/min(len(at),len(bt)) if at and bt else 0.,jac(ag,bg),ratio_len(a,b),not(a and b),nui/(len(num)+len(nums)-nui) if num or nums else 0.,
        bool(num and nums and not nui),nui,bool(pci),bool(pc and pcs and not pci),
        sc==tc,qcountry==tcountry,1/max(rank,1),score,int(passes).bit_count(),
        *[bool(passes&bit) for bit in (1,2,4,8,16,32)],math.log1p(count),nr*ar,nr*(1-ar),ar*(1-nr),
        math.log1p(min(len(nt),len(mt))),math.log1p(min(len(at),len(bt))),
    ]
    assert len(values)==len(NAMES)
    return values


def feature_schema():
    return pa.schema([('qid',pa.int32()),('tid',pa.string()),('source',pa.int8()),('label',pa.int8()),
                      ('split',pa.string()),('fold',pa.int8()),('group_id',pa.int32()),('matches',pa.int16()),
                      ('country',pa.string())]+[(name,pa.float32()) for name in NAMES])


def _feature_table(rows):
    schema = feature_schema()
    cols = [[] for _ in schema]
    for row in rows:
        qid,tid,source,rank,score,passes,count,qn,qa,qc,tn,ta,tc,label,spl,fold_id,gid,matches=row
        values=[qid,tid,source,label,spl,fold_id,gid,matches,qc]+pair_features(qn,qa,qc,tn,ta,tc,rank,score,passes,count)
        for col,v in zip(cols,values): col.append(v)
    return pa.Table.from_arrays([pa.array(v,type=f.type) for f,v in zip(schema,cols)],schema=schema)


def features(ctx,split,tag='main'):
    candidates=ctx.root/'cache'/'candidates'/split/tag
    manifest=read_json(candidates/'manifest.json')
    out=ctx.root/'cache'/'features'/split/tag
    out.mkdir(parents=True,exist_ok=True)
    db=ctx.db()
    workers = int(ctx.cfg.get('feature_workers', 1))
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    for path in sorted(candidates.glob('part-*.parquet')):
        dest=out/path.name
        signature=fingerprint({'candidate':read_json(path.with_suffix('.json'))['signature'],'features':NAMES})
        marker=dest.with_suffix('.json')
        if marker.exists():
            if read_json(marker)['signature']!=signature:
                raise ValueError('Feature schema changed; use a new tag')
            continue
        ctx.guard_disk()
        label="CASE WHEN t.tid IS NULL THEN 0 ELSE 1 END" if split=='train' else '-1'
        truth_join="LEFT JOIN truth t ON c.qid=t.qid AND r.entity_id=t.tid" if split=='train' else ''
        split_columns="s.split,s.fold,s.group_id,s.matches" if split=='train' else "'test',-1,-1,-1"
        split_join="JOIN splits s ON s.qid=c.qid" if split=='train' else ''
        sql=f"""SELECT c.qid,r.entity_id,c.source,c.rank,c.retrieval_score,c.passes,
            count(*) OVER(PARTITION BY c.qid) AS nc,
            q.business_name,q.business_address,q.country,r.business_name,r.business_address,r.country,
            {label} AS label,{split_columns}
            FROM read_parquet('{sqlpath(path)}') c
            JOIN {split}_s1 q ON q.rid=c.qid
            JOIN (SELECT *,2 AS source FROM {split}_s2 UNION ALL SELECT *,3 AS source FROM {split}_s3) r
              ON c.rid=r.rid AND c.source=r.source {truth_join} {split_join}
            ORDER BY c.qid,c.source,c.rank"""
        cursor=db.execute(sql)
        tmp=dest.with_suffix('.tmp')
        writer=None
        total=0
        schema=feature_schema()
        writer=pq.ParquetWriter(tmp,schema,compression='zstd')
        try:
            for rows in batches(cursor,10000):
                if pool:
                    table = pa.concat_tables(list(pool.map(_feature_table,
                        [rows[i:i+500] for i in range(0,len(rows),500)])))
                else:
                    table = _feature_table(rows)
                writer.write_table(table)
                total+=len(rows)
        finally:
            writer.close()
        os.replace(tmp,dest)
        write_json(marker,{'signature':signature,'pairs':total,'qids':read_json(path.with_suffix('.json'))['qids']})
        ctx.event('features',split=split,tag=tag,shard=path.name,pairs=total)
    if pool:
        pool.shutdown()
    db.close()
    write_json(out/'manifest.json',{'candidate_manifest':manifest,'features':NAMES,'complete':True})
    return out


def matrix(table):
    return np.column_stack([table[n].to_numpy() for n in NAMES]).astype(np.float32,copy=False)


def baseline_score(table):
    n=table['name_ratio'].to_numpy();a=table['address_ratio'].to_numpy()
    exact=table['name_exact'].to_numpy()*table['address_exact'].to_numpy()
    conflict=table['number_conflict'].to_numpy()
    return np.maximum(exact,np.where((n>=.93)&(a>=.88)&(conflict==0),.9,0)).astype(np.float32)
