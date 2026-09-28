from __future__ import annotations

import csv
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

from .common import digest, read_json, sqlpath, write_json


def validate(ctx):
    db=ctx.db();issues=[]
    for table,filename,column in [('matches','matching_results.tsv','matched_entity_ids'),('candidates','candidate_pairs.tsv','candidate_entity_ids')]:
        path=ctx.root/'output'/filename
        if not path.exists():raise FileNotFoundError(path)
        # Strict structural scan catches tabs, quoting, whitespace and blank IDs.
        with open(path,encoding='utf-8',newline='') as f:
            if f.readline().rstrip('\r\n')!=f'source1_entity_id\t{column}':raise ValueError('Invalid header: '+filename)
            for line_no,line in enumerate(f,2):
                fields=line.rstrip('\r\n').split('\t')
                if len(fields)!=2 or not fields[0] or fields[0].strip()!=fields[0]:
                    raise ValueError(f'{filename}:{line_no}: malformed row')
                ids=fields[1].split(',') if fields[1] else []
                if any(not x or x.strip()!=x or not x.startswith(('S2-','S3-')) for x in ids):
                    raise ValueError(f'{filename}:{line_no}: malformed ID list')
                if len(ids)!=len(set(ids)):raise ValueError(f'{filename}:{line_no}: duplicate IDs')
        db.execute(f"CREATE OR REPLACE TEMP TABLE {table} AS SELECT source1_entity_id AS qid,coalesce({column},'') AS ids FROM read_csv('{sqlpath(path)}',delim='\t',header=true,all_varchar=true,quote='')")
        checks={
            'duplicate_rows':f'SELECT count(*)-count(DISTINCT qid) FROM {table}',
            'missing_rows':f'SELECT count(*) FROM test_s1 q ANTI JOIN {table} r ON q.entity_id=r.qid',
            'unknown_reference':f'SELECT count(*) FROM {table} r ANTI JOIN test_s1 q ON r.qid=q.entity_id',
        }
        db.execute(f"CREATE OR REPLACE TEMP TABLE {table}_pairs AS SELECT qid,tid FROM {table},unnest(string_split(ids,',')) x(tid) WHERE tid<>''")
        checks['unknown_target']=f"SELECT count(*) FROM {table}_pairs p ANTI JOIN (SELECT entity_id FROM test_s2 UNION ALL SELECT entity_id FROM test_s3) r ON p.tid=r.entity_id"
        for key,sql in checks.items():
            n=db.execute(sql).fetchone()[0]
            if n:issues.append(dict(file=filename,issue=key,count=n))
    outside=db.execute('SELECT count(*) FROM matches_pairs m ANTI JOIN candidates_pairs c USING(qid,tid)').fetchone()[0]
    if outside:issues.append(dict(issue='matches_not_candidates',count=outside))
    expected=db.execute('SELECT count(*) FROM test_s1').fetchone()[0]
    db.close()
    result=dict(pass_validation=not issues,expected_rows=expected,issues=issues,
                hashes={p.name:digest(p) for p in (ctx.root/'output').glob('*.tsv')})
    write_json(ctx.root/'reports'/'validation.json',result)
    if issues:raise ValueError(f'Strict validation failed: {issues}')
    # The official validator stores all candidate sets in Python memory. Run it on
    # bounded S1 partitions while retaining the full S2/S3 ID universe.
    official=ctx.root/'data'/'utils'/'validate_submission.py'
    if official.exists():
        try:
            run_official_partitioned(ctx,official)
            result['official_validator_passed']=True
        except Exception as exc:
            result.update(pass_validation=False,official_validator_passed=False,official_error=str(exc))
            write_json(ctx.root/'reports'/'validation.json',result)
            raise
    write_json(ctx.root/'reports'/'validation.json',result)
    ctx.event('validate',expected_rows=expected,passed=True)
    return result


def run_official_partitioned(ctx,script):
    scratch=ctx.root/'temp'/'official-validation';scratch.mkdir(parents=True,exist_ok=True)
    # Restrict both source and target universes per partition after the global
    # strict validator established ID membership; avoids a giant Python set.
    spec=importlib.util.spec_from_file_location('official_submission_validator',script)
    official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    db=ctx.db();count=0
    query=db.execute('SELECT rid,entity_id,business_name,business_address,country FROM test_s1 ORDER BY rid')
    from .common import batches
    match=open(ctx.root/'output'/'matching_results.tsv',encoding='utf-8')
    candidate=open(ctx.root/'output'/'candidate_pairs.tsv',encoding='utf-8')
    next(match);next(candidate)
    try:
        for rows in batches(query,2000):
            ids2=set();ids3=set()
            with open(scratch/'test_source1.tsv','w',encoding='utf-8',newline='') as f:
                writer=csv.writer(f,delimiter='\t');writer.writerow(['entity_id','business_name','business_address','country'])
                writer.writerows(r[1:] for r in rows)
            for src,filename,col in [(match,'matching_results.tsv','matched_entity_ids'),(candidate,'candidate_pairs.tsv','candidate_entity_ids')]:
                with open(scratch/filename,'w',encoding='utf-8',newline='') as f:
                    f.write('source1_entity_id\t'+col+'\n')
                    for row in rows:
                        line=next(src);qid,_,rest=line.partition('\t')
                        if qid!=row[1]:raise ValueError('Output order differs from test S1 order')
                        f.write(line)
                        for tid in rest.strip().split(','):
                            if tid.startswith('S2-'):ids2.add(tid)
                            elif tid.startswith('S3-'):ids3.add(tid)
            for source,ids in ((2,ids2),(3,ids3)):
                with open(scratch/f'test_source{source}.tsv','w',encoding='utf-8') as f:
                    f.write('entity_id\n');f.writelines(t+'\n' for t in sorted(ids))
            with contextlib.redirect_stdout(io.StringIO()):
                errors,warnings=official.validate(str(scratch/'matching_results.tsv'),str(scratch/'candidate_pairs.tsv'),str(scratch),check_ids=True)
            if errors or warnings:raise RuntimeError(f'Official validator: errors={errors}; warnings={warnings}')
            count+=1
            if count%100==0:ctx.event('official_validation',partitions=count)
    finally:match.close();candidate.close();db.close()
    write_json(ctx.root/'reports'/'official-validation.json',dict(partitions=count,passed=True,
        note='Full-universe membership checked globally by DuckDB; supplied validator then run with --check-ids on bounded partitions.'))


def package(ctx,team):
    if not re.fullmatch(r'[A-Za-z0-9_-]+',team):raise ValueError('Team name must contain letters, numbers, underscore or hyphen')
    validation=read_json(ctx.root/'reports'/'validation.json')
    if not validation['pass_validation']:raise ValueError('Validate first')
    for name,sha in validation['hashes'].items():
        if digest(ctx.root/'output'/name)!=sha:raise ValueError('Outputs changed since validation')
    source=Path(__file__).resolve().parents[2]
    docs=methodology(ctx,team)
    path=ctx.root/'output'/f'{team}_submission.zip'
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for filename in ('matching_results.tsv','candidate_pairs.tsv'):
            z.write(ctx.root/'output'/filename,'output/'+filename)
        for p in source.rglob('*'):
            if p.is_file() and not any(v in p.parts for v in ('__pycache__','.pytest_cache','business_entity_resolution.egg-info')) and p.suffix not in ('.pyc',):
                z.write(p,'code/business_entity_resolution/'+p.relative_to(source).as_posix())
        for p in (ctx.root/'models').glob('*'):
            if p.is_file():z.write(p,'code/business_entity_resolution/artifacts/models/'+p.name)
        for p in (ctx.root/'reports').glob('*.json'):
            z.write(p,'code/business_entity_resolution/artifacts/reports/'+p.name)
        z.writestr('Documentation_template.md',docs)
    (ctx.root/'output'/'Documentation_template.md').write_text(docs,encoding='utf-8')
    ctx.event('package',path=str(path),bytes=path.stat().st_size)
    return path


def methodology(ctx,team):
    def report(name):
        p=ctx.root/'reports'/name
        return read_json(p) if p.exists() else {'status':'Not measured'}
    prediction=report('prediction.json')
    is_baseline=bool(prediction.get('baseline'))
    model_description=('This package contains a conservative deterministic baseline, not the trained ML matcher. '
        'Candidates are exact normalized name-and-address agreements; all are accepted by that rule.' if is_baseline else
        'LightGBM classifiers and an optional CatBoost challenger use name, address, numeric, script and retrieval features. '
        'Hard negatives come from retrieval; thresholds and a maximum-score singleton gate use grouped out-of-fold macro F0.5.')
    return f'''# ML Challenge 2026: Business Entity Resolution

Team name: {team}
Team members: to be supplied by the submitting team.

## 1. Executive summary
Offline, CPU-based retrieval and pair classification with precision-focused set selection.
All learned business information comes from the supplied labeled records. No external business lookup is used.

## 2. Methodology
Raw Unicode text is preserved. Conservative normalized, core-name and accent-folded views support retrieval.
Connected business/signature groups define development and held-out partitions. IDs are never predictive features.
Country is open-ended and unseen countries follow the complete pipeline.

## 3. Candidate generation
Country/source-partitioned SQLite FTS indexes combine rare tokens, character trigrams, joint name/address evidence,
and exact normalized name-address lookup. Candidate files contain exactly the pairs scored at inference.
The bounded same-day configuration disables character-trigram retrieval passes to reduce runtime;
character similarities remain matching features. The runtime configuration below is authoritative.
Candidate budgets are selected by measured recall. Exact duplicate records are retained without a top-k truncation.

Benchmark:
```json
{json.dumps(report('benchmark.json'),indent=2)}
```

## 4. Matching model
{model_description}
Candidate omissions are not repaired using labels.
Models use MIT/Apache-2.0 licensed frameworks; no pretrained neural models or external identity data are used.

## 5. Results and error analysis
Training scope and runtime configuration:
```json
{json.dumps(report('bounded-run.json'),indent=2)}
```
When the bounded runner is used, the model is trained on a reproducible sample of reference
entities while retrieval searches the complete target corpus. Final retraining includes all
sampled partitions after the locked holdout evaluation; it does not include every training entity.

Prediction mode:
```json
{json.dumps(prediction,indent=2)}
```
Development evaluation:
```json
{json.dumps(report('evaluation.json'),indent=2)}
```
Locked holdout:
```json
{json.dumps(report('holdout.json'),indent=2)}
```
Submission validation:
```json
{json.dumps(report('validation.json'),indent=2)}
```

## 6. Limitations
France has no labeled validation data. Country-transfer tests are proxies, not a France score.
No score or recall target is claimed unless present in measured reports. Common-name truncation and cross-script
name changes can lower recall. Model retraining after threshold selection can shift score calibration.

## Appendix: reproduction
See code/business_entity_resolution/README.md and the pinned requirements.txt. Configuration and reports are included.
The original dataset archive must be supplied locally. Generated caches are rebuildable and excluded from the package.
'''
