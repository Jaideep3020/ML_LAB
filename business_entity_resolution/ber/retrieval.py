from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import itertools
import json
import math
import os
from pathlib import Path
import sqlite3
import time
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq

from .common import batches, fingerprint, read_json, sqlpath, write_json
from .text import address, core_name, fold, fts_tokens, normalize

RETRIEVAL_VERSION = 10


def index_path(ctx, split, source, country):
    name = fingerprint(country)[:16]
    return ctx.root / "cache" / "indexes" / split / f"s{source}-{name}.sqlite"


def connect(path):
    c = sqlite3.connect(path)
    c.execute("PRAGMA cache_size=-32768")
    c.execute("PRAGMA temp_store=FILE")
    return c


def ensure_term_stats(path):
    c = connect(path)
    try:
        if not c.execute("SELECT 1 FROM sqlite_master WHERE name='term_stats'").fetchone():
            # fts5vocab computes document counts from postings on every lookup.
            # Materialize the identical statistics once for indexed point reads.
            c.execute('BEGIN')
            c.execute('CREATE TABLE term_stats(col TEXT,term TEXT,doc INTEGER,PRIMARY KEY(col,term)) WITHOUT ROWID')
            c.execute('INSERT INTO term_stats SELECT col,term,doc FROM vocabulary')
            c.commit()
    finally:
        c.close()


def build_aliases(ctx):
    """Learn token aliases from the supplied training links only.

    Position-aligned name tokens provide a conservative bridge between Latin
    reference names and Indic-script source names. Ambiguous mappings are
    discarded; no external transliteration table is used.
    """
    dest = ctx.root/'cache'/'aliases-v2.json'
    if dest.exists():
        return
    db = ctx.db()
    rows = db.execute("""SELECT q.business_name,r.business_name,r.entity_id
        FROM truth t JOIN train_s1 q ON q.rid=t.qid
        JOIN (SELECT entity_id,business_name FROM train_s2 UNION ALL SELECT entity_id,business_name FROM train_s3) r
        ON r.entity_id=t.tid""").fetchall()
    mappings = {'S2': {}, 'S3': {}}
    counts = {'S2': {}, 'S3': {}}
    # Recover source from the linked ID prefix while preserving source-specific vocabulary.
    for left, right, tid in rows:
        source = tid[:2]
        x = fold(core_name(left)).split(); y = fold(core_name(right)).split()
        if len(x) != len(y):
            continue
        for u,v in zip(x,y):
            if not u or not v or not u.isascii() or v.isascii() or len(u) < 3:
                continue
            counts[source].setdefault(u, Counter())[v] += 1
    for source, table in counts.items():
        for token, variants in table.items():
            total = sum(variants.values()); value, count = variants.most_common(1)[0]
            if total >= 3:
                # Keep several script variants learned from the labels. A
                # single Latin token can legitimately map to Hindi, Bengali,
                # Telugu, Tamil, or Kannada spellings in this corpus.
                values = [v for v,n in variants.most_common(6) if n >= 2 and n/total >= .05]
                if values:
                    mappings[source][token] = values
    write_json(dest, mappings)
    db.close()


def build_indexes(ctx, split):
    db = ctx.db()
    for source in (2, 3):
        countries = [r[0] for r in db.execute(f"SELECT DISTINCT country FROM {split}_s{source} ORDER BY country").fetchall()]
        for country in countries:
            ctx.guard_disk()
            path = index_path(ctx, split, source, country)
            path.parent.mkdir(parents=True, exist_ok=True)
            done = path.with_suffix(".json")
            if done.exists():
                ensure_term_stats(path)
                continue
            c = connect(path)
            c.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS records(rid INTEGER PRIMARY KEY, n TEXT NOT NULL, a TEXT NOT NULL);
                CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(n,a,ng,ag,content='',detail=column);
                CREATE VIRTUAL TABLE IF NOT EXISTS vocabulary USING fts5vocab(search,'col');
            """)
            last = c.execute("SELECT coalesce(max(rid),0) FROM records").fetchone()[0]
            cursor = db.execute(f"SELECT rid,business_name,business_address FROM {split}_s{source} WHERE country=? AND rid>? ORDER BY rid", [country,last])
            start = time.time()
            count = 0
            for rows in batches(cursor, 10000):
                records, documents = [], []
                for rid, name, addr in rows:
                    n, a = fold(core_name(name)), fold(address(addr))
                    records.append((rid,n,a))
                    documents.append((rid,fts_tokens(n),fts_tokens(a),fts_tokens(n,True),fts_tokens(a,True)))
                c.executemany("INSERT INTO records VALUES(?,?,?)", records)
                c.executemany("INSERT INTO search(rowid,n,a,ng,ag) VALUES(?,?,?,?,?)", documents)
                c.commit()
                count += len(rows)
                if count % 100000 == 0:
                    ctx.guard_disk()
                    ctx.event("index", split=split, source=source, country=country, added=count, seconds=time.time()-start)
            c.execute("CREATE INDEX IF NOT EXISTS exact_na ON records(n,a)")
            c.execute("INSERT INTO search(search) VALUES('optimize')")
            c.commit()
            c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            total = c.execute("SELECT count(*) FROM records").fetchone()[0]
            c.close()
            ensure_term_stats(path)
            write_json(done, {"split":split,"source":source,"country":country,"rows":total})
            ctx.event("index_complete",split=split,source=source,country=country,rows=total)
    db.close()
    if split == 'train':
        build_aliases(ctx)


class Retriever:
    def __init__(self, ctx, split):
        self.ctx, self.split = ctx, split
        self.connections = {}
        self.statistics = {}
        alias_path = ctx.root/'cache'/ctx.cfg.get('alias_file','aliases-v2.json')
        self.aliases = read_json(alias_path) if alias_path.exists() else {'S2':{},'S3':{}}
        self.countries = defaultdict(list)
        for p in sorted((ctx.root / "cache" / "indexes" / split).glob("*.json")):
            meta = read_json(p)
            source, country = meta["source"], meta["country"]
            self.connections[source,country] = connect(p.with_suffix(".sqlite"))
            c = self.connections[source,country]
            c.execute(f"PRAGMA cache_size=-{int(ctx.cfg.get('retrieval_cache_kib',32768))}")
            self.statistics[source,country] = 'term_stats' if c.execute("SELECT 1 FROM sqlite_master WHERE name='term_stats'").fetchone() else 'vocabulary'
            self.countries[source].append(country)
        audit_path = ctx.root / "reports" / "audit.json"
        self.cross = ctx.cfg["cross_country"] or (audit_path.exists() and not read_json(audit_path)["country_blocking_safe_on_training"])

    @lru_cache(maxsize=5000)
    def rare(self, source, country, column, text):
        c = self.connections[source,country]
        tokens = fts_tokens(text, column in ("ng","ag")).split()
        if not tokens:
            return ()
        placeholders = ','.join('?' for _ in tokens)
        table = self.statistics[source,country]
        rows = c.execute(f"SELECT term,doc FROM {table} WHERE col=? AND term IN ({placeholders})", [column]+tokens).fetchall()
        return tuple(sorted(rows,key=lambda x:(x[1],x[0]))[:8])

    def candidates(self, name, addr, country, k):
        n, a = fold(core_name(name)), fold(address(addr))
        combined = []
        for source in (2,3):
            available = self.countries[source]
            selected = available if self.cross or not country or country not in available else [country] + ([""] if "" in available and country else [])
            found = {}
            for ctry in selected:
                c = self.connections[source,ctry]
                if n and a:
                    for (rid,) in c.execute("SELECT rid FROM records WHERE n=? AND a=?", [n,a]):
                        found[rid] = [2., 1, True]
                columns = [("n",n),("a",a)]
                if self.ctx.cfg.get('include_grams', True):
                    columns += [("ng",n),("ag",a)]
                ranked = {col:self.rare(source,ctry,col,text) for col,text in columns}
                queries = []
                alias_variants = [self.aliases.get(f'S{source}',{}).get(token, []) for token in n.split()]
                alias_terms = [v for values in alias_variants for v in values]
                if alias_terms:
                    # Prefer records whose full name is represented by one
                    # learned script variant, which prevents common suffix
                    # aliases from flooding the top-k list.
                    for index in range(max(map(len, alias_variants))):
                        chosen = [values[index] for values in alias_variants if len(values)>index]
                        if len(chosen) >= 2:
                            terms = ['x'+v.encode('utf-8').hex() for v in chosen]
                            queries.append((f"n: ({' AND '.join(terms)})", 2))
                for col in ("n","a"):
                    terms = ranked[col]
                    # Rare terms bounded individually; common terms require conjunction.
                    rare = [t for t,df in terms[:3] if df <= self.ctx.cfg["posting_limit"]]
                    if rare:
                        queries.append((f"{col}: ("+' OR '.join(rare)+")",2 if col=='n' else 4))
                    if len(terms)>1:
                        queries.append((f"{col}: ("+terms[0][0]+' AND '+terms[1][0]+")",2 if col=='n' else 4))
                    if self.ctx.cfg.get('robust_queries', False):
                        bit = 2 if col == 'n' else 4
                        for left,right in ((0,2),(1,2),(2,3)):
                            if len(terms)>right:
                                queries.append((f"{col}: ({terms[left][0]} AND {terms[right][0]})",bit))
                        for term,df in terms[:4]:
                            if df <= self.ctx.cfg['posting_limit']:
                                queries.append((f'{col}: {term}',bit))
                for col in ("ng","ag"):
                    if col not in ranked:
                        continue
                    terms = ranked[col]
                    # Multiple rare-gram conjunctions tolerate a damaged part of a string.
                    width = int(self.ctx.cfg.get('gram_conjunction', 2))
                    for offset in (0,width):
                        chosen = terms[offset:offset+width]
                        if len(chosen)==width:
                            queries.append((f"{col}: ("+' AND '.join(t for t,_ in chosen)+")",8 if col=='ng' else 16))
                if ranked['n'] and ranked['a']:
                    queries.append((f"n: {ranked['n'][0][0]} AND a: {ranked['a'][0][0]}",32))
                    if self.ctx.cfg.get('robust_queries', False):
                        for ni,ai in ((0,1),(1,0),(1,1)):
                            if len(ranked['n'])>ni and len(ranked['a'])>ai:
                                queries.append((f"n: {ranked['n'][ni][0]} AND a: {ranked['a'][ai][0]}",32))
                queries = list(dict.fromkeys(queries))
                for query, bit in queries:
                    results = c.execute("SELECT rowid,rank FROM search WHERE search MATCH ? ORDER BY rank LIMIT ?", [query,k]).fetchall()
                    for rank,(rid,_) in enumerate(results,1):
                        item = found.setdefault(rid,[0.,0,False])
                        item[0] += 1. / (rank+10)
                        item[1] |= bit
            ordered = sorted(found.items(),key=lambda x:(not x[1][2],-x[1][0],x[0]))
            for rank,(rid,(score,passes,exact)) in enumerate(ordered,1):
                # Keep the union of independently ranked passes. A strong
                # address-only match must not be evicted by name-only distractors.
                combined.append((rid,source,rank,float(score),passes))
        return combined

    def close(self):
        self.rare.cache_clear()
        for c in self.connections.values():
            c.close()


CANDIDATE_SCHEMA = pa.schema([('qid',pa.int32()),('rid',pa.int32()),('source',pa.int8()),('rank',pa.int32()),('retrieval_score',pa.float32()),('passes',pa.int16())])


_worker_retriever = None


def _start_retriever(root, cfg, split):
    from .common import Context
    global _worker_retriever
    _worker_retriever = Retriever(Context(root, cfg), split)


def _retrieve_chunk(task):
    rows, k = task
    return [(qid,)+p for qid,name,addr,country in rows
            for p in _worker_retriever.candidates(name,addr,country,k)]


def retrieve(ctx, split, limit=0, k=None, tag="main", qsplit=None):
    k = k or ctx.cfg['query_limit']
    folder = ctx.root / "cache" / "candidates" / split / tag
    folder.mkdir(parents=True,exist_ok=True)
    db = ctx.db()
    sql = f"SELECT q.rid,q.business_name,q.business_address,q.country FROM {split}_s1 q"
    params = []
    if qsplit:
        if split!='train':
            raise ValueError('qsplit is only valid for training')
        sql += " JOIN splits s ON q.rid=s.qid WHERE s.split=?"
        params.append(qsplit)
    if limit:
        # A bounded run must sample across the corpus, not take an ID prefix.
        sql += f" ORDER BY hash(q.rid,{int(ctx.cfg['seed'])}) LIMIT {int(limit)}"
        sql = f"SELECT * FROM ({sql}) sampled ORDER BY rid"
    else:
        sql += " ORDER BY q.rid"
    cursor = db.execute(sql,params)
    retriever = Retriever(ctx,split)
    check=db.cursor()
    required={(source,country) for source in (2,3) for (country,) in check.execute(f'SELECT DISTINCT country FROM {split}_s{source}').fetchall()}
    check.close()
    if required-set(retriever.connections):
        retriever.close();db.close()
        raise RuntimeError('Build all target country/source indexes before retrieval')
    signature = fingerprint({"version":RETRIEVAL_VERSION,"cfg":ctx.cfg,"split":split,"limit":limit,"k":k,"qsplit":qsplit,
                             "input":read_json(ctx.root/'reports'/'input.json')['sha256']})
    start = time.time()
    workers = int(ctx.cfg.get('retrieval_workers', 1))
    pool = ProcessPoolExecutor(max_workers=workers, initializer=_start_retriever,
                               initargs=(str(ctx.root),ctx.cfg,split)) if workers > 1 else None
    queries = pairs = 0
    retrieval_seconds = 0.
    for idx,rows in enumerate(batches(cursor,ctx.cfg['batch_size'])):
        path = folder / f"part-{idx:06d}.parquet"
        marker = path.with_suffix('.json')
        if marker.exists():
            meta = read_json(marker)
            if meta['signature']!=signature:
                raise ValueError('Candidate configuration changed; use a new tag')
            queries += meta['queries']; pairs += meta['pairs']
            retrieval_seconds += meta.get('seconds', 0.)
            continue
        ctx.guard_disk()
        shard_start = time.time()
        output = []
        if pool:
            # Larger IPC batches reduce process and SQLite setup overhead while
            # preserving the exact per-query retrieval logic and ordering.
            tasks = [(rows[i:i+64], k) for i in range(0,len(rows),64)]
            for result in pool.map(_retrieve_chunk,tasks):
                output.extend(result)
        else:
            for qid,name,addr,country in rows:
                output.extend((qid,)+p for p in retriever.candidates(name,addr,country,k))
        columns = list(zip(*output)) if output else [[] for _ in CANDIDATE_SCHEMA]
        table = pa.Table.from_arrays([pa.array(v,type=f.type) for f,v in zip(CANDIDATE_SCHEMA,columns)],schema=CANDIDATE_SCHEMA)
        tmp = path.with_suffix('.tmp')
        pq.write_table(table,tmp,compression='zstd')
        os.replace(tmp,path)
        # Include every queried ID, including entities with zero candidates.
        meta = dict(signature=signature,queries=len(rows),pairs=len(output),qids=[r[0] for r in rows],seconds=time.time()-shard_start)
        retrieval_seconds += meta['seconds']
        write_json(marker,meta)
        queries += len(rows); pairs += len(output)
        ctx.event('retrieve',split=split,tag=tag,queries=queries,pairs=pairs,seconds=time.time()-start)
    if pool:
        pool.shutdown()
    retriever.close(); db.close()
    write_json(folder/'manifest.json',dict(signature=signature,queries=queries,pairs=pairs,k=k,limit=limit,qsplit=qsplit,complete=True,retrieval_seconds=retrieval_seconds))
    return folder


def benchmark(ctx, limit=10000):
    reports=[]
    budgets=list(ctx.cfg['candidate_budgets'])
    # One explicit widening step if the planned budgets miss the recall target.
    expanded=max(budgets)*2
    for k in budgets+[expanded]:
        if k==expanded and any(r['recall']>=ctx.cfg['retrieval_recall_target'] for r in reports):
            break
        start=time.time()
        folder=retrieve(ctx,'train',limit,k,tag=f'benchmark-v{RETRIEVAL_VERSION}-k{k}',qsplit='tune')
        db=ctx.db()
        files=str(folder/'part-*.parquet').replace('\\','/')
        db.execute(f"CREATE OR REPLACE TEMP VIEW candidates AS SELECT * FROM read_parquet('{sqlpath(files)}')")
        qids=[]
        for p in folder.glob('part-*.json'):
            qids.extend(read_json(p)['qids'])
        db.register('queries',pa.table({'qid':pa.array(qids,type=pa.int32())}))
        counts=db.execute("""SELECT count(*), count(*) FILTER(WHERE c.qid IS NOT NULL) FROM truth t
            JOIN queries q ON q.qid=t.qid LEFT JOIN
            (SELECT c.qid,r.entity_id FROM candidates c JOIN
             (SELECT rid,entity_id,2 AS source FROM train_s2 UNION ALL SELECT rid,entity_id,3 FROM train_s3) r
             ON c.rid=r.rid AND c.source=r.source) c ON c.qid=t.qid AND c.entity_id=t.tid""").fetchone()
        meta=read_json(folder/'manifest.json')
        report=dict(k=k,queries=len(qids),pairs=meta['pairs'],true_pairs=counts[0],recovered=counts[1],
                    recall=counts[1]/max(1,counts[0]),seconds=meta['retrieval_seconds'],
                    candidate_bytes=sum(p.stat().st_size for p in folder.glob('*.parquet')))
        reports.append(report);db.close()
        ctx.event('benchmark',**report)
    eligible=[r for r in reports if r['recall']>=ctx.cfg['retrieval_recall_target']]
    chosen=min(eligible,key=lambda x:x['k']) if eligible else max(reports,key=lambda x:x['recall'])
    result=dict(results=reports,selected_k=chosen['k'],target_met=bool(eligible),
                warning=None if eligible else 'Recall target not reached; retrieval errors need review before claiming competitive completion.')
    db=ctx.db()
    total_queries=sum(db.execute(f'SELECT count(*) FROM {s}_s1').fetchone()[0] for s in ('train','test'))
    db.close()
    result['projected_queries']=total_queries
    result['projected_candidate_pairs']=round(chosen['pairs']/max(1,chosen['queries'])*total_queries)
    result['projected_candidate_bytes']=round(chosen['candidate_bytes']/max(1,chosen['queries'])*total_queries)
    result['projected_retrieval_hours']=chosen['seconds']/max(1,chosen['queries'])*total_queries/3600
    result['projection_note']='Training benchmark extrapolation; unseen-country density and warm/cold cache differences can change runtime and volume.'
    ctx.guard_disk(result['projected_candidate_bytes'])
    write_json(ctx.root/'reports'/'benchmark.json',result)
    return result
