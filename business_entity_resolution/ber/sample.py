"""Create a clearly marked real-data pilot; never a full submission dataset."""
from __future__ import annotations

import csv
import io
from pathlib import Path
import zipfile


def make_pilot(data_dir,destination,references=5000,distractors=10000):
    data_dir=Path(data_dir);destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    wanted=set()
    queries=set()
    def write(z,name,rows):
        with z.open('student_resource/'+name,'w') as raw:
            text=io.TextIOWrapper(raw,encoding='utf-8',newline='')
            writer=csv.writer(text,delimiter='\t',lineterminator='\n')
            writer.writerows(rows);text.flush();text.detach()
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED) as z:
        def source1(split,limit):
            with open(data_dir/'dataset'/split/f'{split}_source1.tsv',encoding='utf-8',newline='') as f:
                reader=csv.reader(f,delimiter='\t');yield next(reader)
                for i,row in enumerate(reader):
                    if i>=limit:break
                    if split=='train':queries.add(row[0])
                    yield row
        write(z,'dataset/train/train_source1.tsv',source1('train',references))
        def truth():
            with open(data_dir/'dataset'/'train'/'train_ground_truth.tsv',encoding='utf-8',newline='') as f:
                reader=csv.reader(f,delimiter='\t');yield next(reader)
                for row in reader:
                    if row[0] in queries:
                        wanted.update(v for v in row[1].split(',') if v);yield row
        write(z,'dataset/train/train_ground_truth.tsv',truth())
        for source in (2,3):
            def rows(source=source):
                with open(data_dir/'dataset'/'train'/f'train_source{source}.tsv',encoding='utf-8',newline='') as f:
                    reader=csv.reader(f,delimiter='\t');yield next(reader)
                    for i,row in enumerate(reader):
                        if row[0] in wanted or i<distractors:yield row
            write(z,f'dataset/train/train_source{source}.tsv',rows())
        write(z,'dataset/test/test_source1.tsv',source1('test',min(1000,references)))
        for source in (2,3):
            def rows(source=source):
                with open(data_dir/'dataset'/'test'/f'test_source{source}.tsv',encoding='utf-8',newline='') as f:
                    reader=csv.reader(f,delimiter='\t');yield next(reader)
                    for i,row in enumerate(reader):
                        if i>=distractors:break
                        yield row
            write(z,f'dataset/test/test_source{source}.tsv',rows())
        for name in ('README.md','Documentation_template.md','utils/validate_submission.py'):
            p=data_dir/name
            if p.exists():z.write(p,'student_resource/'+name)
        z.writestr('student_resource/PILOT_ONLY.txt','Reduced distractor universe. Metrics are optimistic and are not full-corpus estimates. Test outputs are NOT competition submissions.')
    return destination
