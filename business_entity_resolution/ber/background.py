"""Run the long computation with a durable log and human-readable status."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .common import write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',required=True)
    p.add_argument('--config',required=True)
    p.add_argument('--archive',required=True)
    p.add_argument('--deliverables',required=True)
    args=p.parse_args()
    root=Path(args.run_dir).resolve();dest=Path(args.deliverables).resolve()
    root.mkdir(parents=True,exist_ok=True);dest.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-X','utf8','-m','ber.cli','--run-dir',str(root),'--config',str(Path(args.config).resolve()),
             'run','--archive',str(Path(args.archive).resolve())]
    state=dict(status='starting',run_dir=str(root),pid=os.getpid(),archive=str(Path(args.archive).resolve()))
    def publish():
        state['updated_utc']=datetime.now(timezone.utc).isoformat()
        write_json(dest/'RUN_STATUS.json',state)
        message=(f"# Full dataset run\n\nStatus: **{state['status']}**\n\n"
                 f"Updated: {state['updated_utc']}\n\n"
                 f"Current stage: {state.get('stage','starting')}\n\n"
                 "The separate baseline submission is not the competitive ML result.\n\n")
        details={k:state[k] for k in ('queries','pairs','rows','source','country','trial','rss_gb','peak_rss_gb','error') if k in state}
        if details:message+='Latest progress:\n\n'+''.join(f'- {k}: {v}\n' for k,v in details.items())
        if state['status']=='complete':message+='\nThe full ML submission files are available in this folder.\n'
        elif state['status']=='failed':message+='\nThe run stopped with an error. The durable log and existing stage checkpoints are preserved.\n'
        tmp=dest/'RUN_STATUS.md.tmp';tmp.write_text(message,encoding='utf-8');os.replace(tmp,dest/'RUN_STATUS.md')
    publish()
    with open(root/'run.log','a',encoding='utf-8') as log:
        child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        state.update(status='running',child_pid=child.pid);publish()
        for line in child.stdout:
            log.write(line);log.flush()
            try:
                event=json.loads(line)
                if isinstance(event,dict) and 'stage' in event:
                    state.update(event);publish()
            except json.JSONDecodeError:
                if line.strip():state['last_log_line']=line.strip()[:1000]
        code=child.wait()
    state['exit_code']=code
    if code==0:
        for path in (root/'output').iterdir():
            if path.is_file():shutil.copy2(path,dest/path.name)
        reports=dest/'reports';reports.mkdir(exist_ok=True)
        for path in (root/'reports').glob('*.json'):shutil.copy2(path,reports/path.name)
        state['status']='complete'
    else:
        state.update(status='failed',error=state.get('last_log_line','See run.log'))
    publish()
    raise SystemExit(code)


if __name__=='__main__':main()
