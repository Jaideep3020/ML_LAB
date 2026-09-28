from __future__ import annotations

import argparse
from pathlib import Path

from .common import Context


def main():
    p=argparse.ArgumentParser(description='Offline business entity resolution')
    p.add_argument('--run-dir',default='run')
    p.add_argument('--config',default=str(Path(__file__).resolve().parents[2]/'config.json'))
    sub=p.add_subparsers(dest='command',required=True)
    x=sub.add_parser('prepare');x.add_argument('--archive',required=True)
    sub.add_parser('audit');sub.add_parser('split')
    x=sub.add_parser('index');x.add_argument('--dataset',choices=['train','test'],required=True)
    x=sub.add_parser('baseline');x.add_argument('--dataset',choices=['train','test'],required=True)
    x=sub.add_parser('benchmark');x.add_argument('--limit',type=int,default=10000)
    for command in ('retrieve','features','predict'):
        x=sub.add_parser(command);x.add_argument('--dataset',choices=['train','test'],required=True);x.add_argument('--tag',default='main')
        if command=='retrieve':
            x.add_argument('--limit',type=int,default=0);x.add_argument('--k',type=int);x.add_argument('--query-split',choices=['fit','tune','holdout'])
        if command=='predict':
            x.add_argument('--model',default='selected');x.add_argument('--baseline',action='store_true')
    x=sub.add_parser('train');x.add_argument('--tag',default='main');x.add_argument('--final',action='store_true');x.add_argument('--search',action='store_true')
    x=sub.add_parser('evaluate');x.add_argument('--tag',default='main');x.add_argument('--holdout',action='store_true')
    x=sub.add_parser('transfer');x.add_argument('--tag',default='main')
    x=sub.add_parser('errors');x.add_argument('--tag',default='main')
    x=sub.add_parser('mine');x.add_argument('--scores',required=True)
    sub.add_parser('validate')
    x=sub.add_parser('package');x.add_argument('--team',default='team')
    x=sub.add_parser('run');x.add_argument('--archive',required=True);x.add_argument('--baseline',action='store_true')
    args=p.parse_args();ctx=Context(args.run_dir,args.config)
    from .data import prepare,audit,make_splits
    from .retrieval import build_indexes,retrieve,benchmark
    from .features import features
    if args.command=='prepare': prepare(ctx,args.archive)
    elif args.command=='audit': audit(ctx)
    elif args.command=='split': make_splits(ctx)
    elif args.command=='index': build_indexes(ctx,args.dataset)
    elif args.command=='baseline':
        from .baseline import exact_baseline
        exact_baseline(ctx,args.dataset)
    elif args.command=='benchmark': benchmark(ctx,args.limit)
    elif args.command=='retrieve': retrieve(ctx,args.dataset,args.limit,args.k,args.tag,args.query_split)
    elif args.command=='features': features(ctx,args.dataset,args.tag)
    elif args.command in ('transfer','errors','mine'):
        from .experiments import country_transfer,error_analysis,mine_negatives
        if args.command=='transfer':country_transfer(ctx,args.tag)
        elif args.command=='errors':error_analysis(ctx,args.tag)
        else:mine_negatives(ctx,args.scores)
    elif args.command in ('train','evaluate','predict'):
        from .model import train,evaluate,predict
        if args.command=='train': train(ctx,args.tag,final=args.final,search=args.search)
        elif args.command=='evaluate': evaluate(ctx,args.tag,holdout=args.holdout)
        else: predict(ctx,args.dataset,args.tag,args.model,args.baseline)
    elif args.command in ('validate','package'):
        from .delivery import validate,package
        if args.command=='validate': validate(ctx)
        else: package(ctx,args.team)
    elif args.command=='run':
        from .model import train,evaluate,predict
        from .delivery import validate,package
        prepare(ctx,args.archive);audit(ctx);make_splits(ctx)
        build_indexes(ctx,'train');report=benchmark(ctx)
        k=report['selected_k']
        # Measure feature storage on the selected benchmark before a full run.
        from .retrieval import RETRIEVAL_VERSION
        from .common import read_json,write_json
        btag=f'benchmark-v{RETRIEVAL_VERSION}-k{k}'
        bench_features=features(ctx,'train',btag)
        size=sum(p.stat().st_size for p in bench_features.glob('*.parquet'))
        queries=read_json(ctx.root/'cache'/'candidates'/'train'/btag/'manifest.json')['queries']
        report['projected_feature_bytes']=int(size/max(1,queries)*report['projected_queries'])
        ctx.guard_disk(report['projected_feature_bytes']+report['projected_candidate_bytes'])
        write_json(ctx.root/'reports'/'benchmark.json',report)
        retrieve(ctx,'train',k=k);features(ctx,'train')
        train(ctx,search=not args.baseline);evaluate(ctx)
        from .experiments import country_transfer,error_analysis
        country_transfer(ctx);error_analysis(ctx);evaluate(ctx,holdout=True)
        train(ctx,final=True)
        build_indexes(ctx,'test');retrieve(ctx,'test',k=k);features(ctx,'test')
        predict(ctx,'test',baseline=args.baseline);validate(ctx);package(ctx,'team')


if __name__=='__main__':
    main()
