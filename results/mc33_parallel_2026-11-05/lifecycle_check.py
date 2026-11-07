"""Actual SIGKILL/resume integration check on three isolated historical replays."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from diagnostics.mc33_parallel import freeze, read_json, digest, publish, require, run
from diagnostics.chernov_batch_control import lock_is_held
from diagnostics.audit_mc33_parallel import verify_replay

OUT = Path(__file__).resolve().parent
plan = read_json(OUT/'benchmark_w1/plan.json')
jobs = [j for j in plan['jobs'] if (j['equation_id'], j['episode']) in {
    ('AMV1_F0028_S03', 0), ('AMV1_F0004_S04', 0), ('AMV1_F0004_S04', 1)}]
require(len(jobs) == 3, 'wrong lifecycle scope')
folder = OUT/'lifecycle'
freeze(folder, jobs, plan['data_dir'], plan['protected_sha256'],
       purpose='three historical replays only; interruption/resume integration check')
command = [sys.executable, '-B', str(ROOT/'diagnostics/mc33_parallel.py'), 'run',
           '--out', str(folder), '--workers', '1', '--resume']
with (folder/'killed_controller.log').open('x') as log:
    child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic()+180
        while True:
            require(child.poll() is None, 'controller exited before interruption')
            ready = list((folder/'records').glob('*.json'))
            requests = list((folder/'attempts').glob('*.request.json'))
            states = [read_json(p) for p in (folder/'workers').glob('*.json') if '.warmup.' not in p.name]
            if len(ready) == 1 and len(requests) == 2 and any(
                    s.get('phase') == 'solving' and s.get('key') != ready[0].stem for s in states):
                break
            require(time.monotonic() < deadline, 'integration test did not reach interruption point')
            time.sleep(.1)
        preserved = {str(p):digest(p) for p in ready}
        time.sleep(.5)
        child.kill(); child.wait(timeout=5)
    finally:
        if child.poll() is None:
            child.terminate(); child.wait(timeout=10)

deadline = time.monotonic()+15
while True:
    guardians = [read_json(p) for p in (folder/'control').rglob('*.guardian.json')]
    if guardians and all(g.get('finished') for g in guardians) and not lock_is_held(folder/'batch.lock'):
        break
    require(time.monotonic() < deadline, 'guardian cleanup failed')
    time.sleep(.1)
require(all(g['cleanup_success'] and g['reason'] == 'controller_pipe_closed' for g in guardians), 'bad cleanup')
require(len(list((folder/'records').glob('*.json'))) == 1, 'unexpected extra result')
resumed = run(folder, workers=2, resume=True)
require(resumed['new_terminal'] == 2 and resumed['pending'] == 0, 'wrong resume work')
require(all(digest(Path(p)) == h for p,h in preserved.items()), 'completed record was rewritten')
verify_replay(folder)
before = {str(p):digest(p) for p in (folder/'records').glob('*.json')}
again = run(folder, workers=8, resume=True)
require(again['worker_startups'] == 0, 'completed resume started workers')
require(all(digest(Path(p)) == h for p,h in before.items()), 'complete resume changed results')
requests = [read_json(p) for p in (folder/'attempts').glob('*.request.json')]
by_key = {}
for r in requests:
    by_key.setdefault(r['job']['key'], []).append(r)
replayed = [rs for rs in by_key.values() if len(rs) == 2]
require(len(requests) == 4 and len(replayed) == 1 and replayed[0][0]['job'] == replayed[0][1]['job'], 'retry identity changed')
publish(OUT/'lifecycle_verification.json', dict(passed=True, killed_controller_returncode=child.returncode,
    guardians=guardians, preserved_records=preserved, completed_replays=3,
    interrupted_physical_attempts=1, total_physical_requests=4,
    resumed_with_workers=2, completed_resume=again, same_seed_retry_verified=True))
print('Lifecycle SIGKILL/resume: passed')
