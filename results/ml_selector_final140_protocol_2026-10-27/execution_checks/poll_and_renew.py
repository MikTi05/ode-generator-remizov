"""One externally invoked inspection/renewal; no loop, solver or model use."""
import collections,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
out=Path(__file__).resolve().parent.parent
read=lambda p:json.loads(p.read_text())
active=read(out/'active.json') if (out/'active.json').exists() else {}
renew=None
if active and not active.get('finished'):
 q=subprocess.run([str(ROOT/'.venv/bin/python'),'-B',str(ROOT/'diagnostics/run_chernov_batch.py'),'--output-dir',str(out),'--keep-alive'],capture_output=True,text=True)
 renew=dict(returncode=q.returncode,stdout=q.stdout.strip(),stderr=q.stderr.strip())
records=[read(p) for p in (out/'records').glob('*.json')]
budget=read(out/'budget.json') if (out/'budget.json').exists() else {}
workers=[read(p) for p in (out/'workers').glob('*.json') if not p.name.endswith('.warmup.json')]
workers.sort(key=lambda r:r.get('monotonic',0))
profiles={p:dict(records=sum(r['profile']==p for r in records),accurate=sum(r['profile']==p and r.get('accuracy_met') is True for r in records),violations=sum(r['profile']==p and r.get('accuracy_met') is False for r in records)) for p in ('R128','R256','MC1')}
lease=read(Path(active['control_dir'])/'lease.json') if active else {}
guards=[read(p) for p in Path(active['control_dir']).glob('*.guardian.json')] if active else []
progress=read(out/'progress.json') if (out/'progress.json').exists() else {}
row=dict(observed_epoch=time.time(),observed_monotonic=time.monotonic(),external_call=True,renewal=renew,
 records=len(records),requested=len(list((out/'events').glob('*.request.json'))),statuses=dict(collections.Counter(r['status'] for r in records)),profiles=profiles,
 elapsed_seconds=time.time()-budget['started_epoch'] if budget else None,remaining_seconds=budget.get('deadline_epoch',time.time())-time.time(),
 lease_age_seconds=time.monotonic()-lease['renewed_monotonic'] if lease else None,
 active_finished=active.get('finished'),worker=workers[-1] if workers else None,
 guardian=[{k:g.get(k) for k in ('finished','cleanup_success','reason')} for g in guards],thermal=progress.get('thermal'))
with (out/'execution_checks/external_owner.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
print(json.dumps(row),flush=True)
if renew and renew['returncode'] and not all(g.get('finished') for g in guards):raise SystemExit(1)
