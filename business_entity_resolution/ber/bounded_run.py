"""Full-corpus matching with a bounded, reproducibly sampled training set."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import time

from .common import Context, read_json, write_json, sqlpath


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--deliverables', required=True)
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--training-queries', type=int, default=50000)
    parser.add_argument('--team', default='team')
    args = parser.parse_args()
    ctx = Context(args.run_dir, args.config)
    dest = Path(args.deliverables).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    from .retrieval import RETRIEVAL_VERSION
    tag = f'bounded-v{RETRIEVAL_VERSION}-n{args.training_queries}-k{args.k}'
    state = dict(status='running', tag=tag, training_queries=args.training_queries,
                 k=args.k, started_utc=datetime.now(timezone.utc).isoformat())

    def publish(stage, **extra):
        state.update(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
        write_json(dest/'RUN_STATUS.json', state)
        (dest/'RUN_STATUS.md').write_text(
            '# Amazon entity resolution run\n\n'
            f"Status: {state['status']}\n\nStage: {stage}\n\n"
            f"Updated: {state['updated_utc']}\n\n"
            f'Training uses up to {args.training_queries:,} sampled references and the full target corpus. '
            'Test predictions cover every reference, including France.\n\n'
            + (f"Progress: {state['progress']}\n\n" if 'progress' in state else '')
            + (f"Error: {state['error']}\n" if 'error' in state else ''), encoding='utf-8')

    original_event = ctx.event
    def progress_event(event_stage, **details):
        original_event(event_stage, **details)
        progress = ', '.join(f'{key}: {details[key]}' for key in
                             ('queries','pairs','shard','country','trial','partitions') if key in details)
        if progress:
            publish(state.get('stage', event_stage), progress=progress)
    ctx.event = progress_event

    def stage(name, action):
        publish(name)
        started = time.time()
        result = action()
        ctx.event('bounded_stage_complete', name=name, seconds=time.time()-started)
        return result

    try:
        from .retrieval import retrieve, build_indexes
        from .features import features
        from .model import train, evaluate, predict
        from .experiments import country_transfer, error_analysis
        from .delivery import validate, package
        write_json(ctx.root/'reports'/'execution-plan.json', dict(
            k=args.k, training_queries=args.training_queries, config=ctx.cfg,
            rationale='Bounded model search and word retrieval selected for same-day completion; 99% recall target not met.',
            scope='All target records; sampled training references; every test reference.'))

        # Screening features already exist when continuing a measured run.
        screening = ctx.root/'cache'/'features'/'train'/f'benchmark-v{RETRIEVAL_VERSION}-k{args.k}'
        if (screening/'manifest.json').exists():
            measured = read_json(screening/'manifest.json')['candidate_manifest']['queries']
            measured_bytes = sum(p.stat().st_size for p in screening.glob('*.parquet'))
            db = ctx.db()
            test_count = db.execute('SELECT count(*) FROM test_s1').fetchone()[0]
            db.close()
            projected = measured_bytes/max(1,measured)*(test_count+args.training_queries)
            # Leave space for joins, score files, sorting, final TSVs and ZIP.
            ctx.guard_disk(int(projected*4))
            write_json(ctx.root/'reports'/'storage-plan.json', dict(
                projected_feature_bytes=int(projected), guarded_bytes=int(projected*4),
                screening_queries=measured, training_query_limit=args.training_queries,
                test_queries=test_count, note='Extrapolation; unseen-country density can differ.'))

        def save_row_ids():
            # Older prepared checkpoints may predate sorted-ID assignment. Ship
            # their mapping so a fresh preparation reproduces the same splits.
            folder = Path(__file__).resolve().parents[2]/'artifacts'/'row_ids'
            folder.mkdir(parents=True, exist_ok=True)
            provenance = {'archive_sha256': read_json(ctx.root/'reports'/'input.json')['sha256']}
            marker = folder/'manifest.json'
            if marker.exists() and read_json(marker) != provenance:
                raise ValueError('Source package already contains row IDs for another archive')
            db = ctx.db()
            try:
                for split in ('train', 'test'):
                    for source in (1, 2, 3):
                        table = f'{split}_s{source}'
                        target = folder/f'{table}.parquet'
                        tmp = target.with_suffix('.tmp')
                        db.execute(f"COPY (SELECT rid,entity_id FROM {table} ORDER BY rid) TO '{sqlpath(tmp)}' (FORMAT PARQUET, COMPRESSION ZSTD)")
                        tmp.replace(target)
            finally:
                db.close()
            write_json(marker, provenance)

        stage('reproducibility metadata', save_row_ids)

        stage('training candidates', lambda: retrieve(ctx, 'train', args.training_queries, args.k, tag))
        stage('training features', lambda: features(ctx, 'train', tag))
        stage('model selection', lambda: train(ctx, tag, search=True))
        stage('development evaluation', lambda: evaluate(ctx, tag))
        stage('country transfer', lambda: country_transfer(ctx, tag))
        stage('error analysis', lambda: error_analysis(ctx, tag))
        stage('locked holdout', lambda: evaluate(ctx, tag, holdout=True))
        stage('final model', lambda: train(ctx, tag, final=True))
        stage('test indexes', lambda: build_indexes(ctx, 'test'))
        stage('test candidates', lambda: retrieve(ctx, 'test', k=args.k, tag=tag))
        stage('test features', lambda: features(ctx, 'test', tag))
        stage('full test prediction', lambda: predict(ctx, 'test', tag))
        stage('submission validation', lambda: validate(ctx))
        write_json(ctx.root/'reports'/'bounded-run.json', dict(
            training_reference_limit=args.training_queries, tag=tag, k=args.k,
            target_corpus='complete', test_reference_coverage='complete', config=ctx.cfg,
            note='Final training uses all retrieved sampled references, not all training references.'))
        archive = stage('submission packaging', lambda: package(ctx, args.team))
        for filename in ('matching_results.tsv', 'candidate_pairs.tsv', 'Documentation_template.md', archive.name):
            shutil.copy2(ctx.root/'output'/filename, dest/filename)
        reports = dest/'reports'
        reports.mkdir(exist_ok=True)
        for file in (ctx.root/'reports').glob('*.json'):
            shutil.copy2(file, reports/file.name)
        publish('complete', status='complete')
    except Exception as exc:
        publish(state.get('stage', 'starting'), status='failed', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
