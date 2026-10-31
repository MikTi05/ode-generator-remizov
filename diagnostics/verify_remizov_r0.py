"""Independent saved-only R0 and label audit, including protected-file hashes."""
from __future__ import annotations
import argparse,csv,json,sys
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from diagnostics.mc_pilot_p2 import digest,read_json,write_json,snapshot
from diagnostics.chernov_batch_control import lock_is_held


def main(out):
    from all_methods_dataset_loader import load_all_methods_equations
    plan=read_json(out/'plan.json');before=read_json(out/'protected_before.json')
    after=snapshot(out);changed={k for k in before if after.get(k)!=before[k]}
    assert changed<= {'PROJECT_STATUS.md'},changed
    additions=set(after)-set(before)
    assert additions<= {'diagnostics/report_remizov_r0.py','diagnostics/verify_remizov_r0.py','diagnostics/resume_remizov_r0.py','tests/test_remizov_r0_resume.py'},additions
    assert not lock_is_held(ROOT/'results/.chernov_batch_controller.lock')
    assert not lock_is_held(ROOT/'results/.chernov_batch_compute.lock')
    assert read_json(out/'active.json')['finished']
    assert not read_json(out/'pending.json')['keys']
    records=list((out/'records').glob('*.json'))
    requests=[read_json(p) for p in (out/'events').glob('*.request.json')]
    assert len(records)==38 and len(requests)==39
    repeats={k:n for k,n in Counter(r['job']['key'] for r in requests).items() if n>1}
    assert repeats=={'AMV1_F0075_S00_R256_stage0_episode0':2}
    assert all(read_json(p)['status']=='completed' for p in records)
    contract=read_json(ROOT/'reports/ml_selector_contract/plan.json')
    data=ROOT/contract['data_dir']
    eqs={e.equation_id:e for e in load_all_methods_equations(data/'all_methods_equations_v1.csv',data/'all_methods_equations_v1_metadata.jsonl')}
    manifest=list(csv.DictReader((ROOT/'reports/ml_selector_contract/pilot_manifest.csv').open()))
    ids={r['equation_id'] for r in manifest};assert len(ids)==40
    errors={};numeric_checks=0
    for pid in ids:
        errors[pid]={}
        for profile in ('R128','R256'):
            p=out/'records'/f'{pid}_{profile}_stage0_episode0.json'
            if not p.exists():
                p=(ROOT/'results/joint_grid100_2026-10-25/records'/f'{pid}_R.json') if profile=='R128' else (ROOT/'results/final_grid5_2026-10-25/records'/f'{pid}_R256.json')
            r=read_json(p);z=r['result'];exact=np.asarray(eqs[pid].y_exact(np.asarray(contract['target_x'])))
            assert z['problem_id']==pid and z['n']==int(profile[1:]) and z['success']
            assert np.array_equal(z['x'],contract['target_x']) and np.array_equal(r['y_exact'],exact)
            assert r['expression_hash']==eqs[pid].metadata['expression_hash']
            err=float(np.max(abs(np.asarray(z['f'])-exact)))
            assert err==r.get('E_grid',r.get('E_inf'))
            errors[pid][profile]=err;numeric_checks+=1
    rcount={p:sum(errors[pid][p]<=.01 for pid in ids) for p in ('R128','R256')}
    selected='R256' if rcount['R256']>rcount['R128'] else 'R128'
    analysis=read_json(out/'analysis.json');assert rcount==analysis['R_accurate'] and selected==analysis['selected_R_profile']
    labels=read_json(out/'selected_labels_1000.json')['rows'];assert len(labels)==1000
    index={x['equation_id']:x for x in labels};assert set(index)==set(eqs)
    p1=ROOT/'reports/ml_selector_contract/sentinel_p1_2026-10-26/records'
    p2=ROOT/'reports/ml_selector_contract/pilot_p2_2026-10-26/records'
    outcome=Counter();mc_checks=0
    for pid in ids:
        p=next(iter(p1.glob(f'{pid}_MC1_stage1_episode0.json')),None)
        if p is None:p=next(iter(p2.glob(f'{pid}_MC1_stage2_episode0.json')),None)
        assert p is not None
        r=read_json(p);z=r['result'];exact=np.asarray(eqs[pid].y_exact(np.asarray(contract['target_x'])))
        assert r['status']=='completed' and z['success'] and np.array_equal(z['x'],contract['target_x'])
        mcerr=float(np.max(abs(np.asarray(z['f'])-exact)));assert mcerr==r['E_grid']
        rp=errors[pid][selected]<=.01;mp=mcerr<=.01
        want='unknown' if rp and mp else ('remizov' if rp else ('monte_carlo' if mp else 'none'))
        assert index[pid]['label']==want,(pid,want,index[pid]['label'])
        outcome[want]+=1;mc_checks+=1
    assert all(x['label']=='unknown' for x in labels if x['equation_id'] not in ids)
    assert outcome['monte_carlo']==0
    result=dict(passed=True,protected_before_files=len(before),protected_changed=sorted(changed),allowed_new_code=sorted(additions),
        new_R0_records=len(records),new_R0_requests=39,interrupted_attempts=1,all_R_cells=numeric_checks,
        independent_MC_episode0_errors=mc_checks,R_accurate=rcount,selected_R=selected,
        pilot_labels=dict(outcome),all1000_labels=dict(Counter(x['label'] for x in labels)),
        no_solver_or_fit_in_verifier=True)
    write_json(out/'independent_verification.json',result,exclusive=True)
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);main(ap.parse_args().out.resolve())
