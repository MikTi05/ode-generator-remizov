"""Independent verification of a stopped validation preflight; no solver/fit."""
from __future__ import annotations
import argparse,math,statistics,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from diagnostics.train_method_selector import write,snapshot
from ml.selector_evidence import read,sha,CONTRACT
MODEL=ROOT/'results/ml_probability_selector_2026-10-26'


def verify(out):
    from ml.remizov_features import load_inputs
    from ml.probabilistic_selector import ProbabilisticSelector,estimates_from_heads,choose
    plan=read(out/'plan.json');budget=read(out/'budget.json');pred=read(out/'frozen_predictions.json')
    assert plan['status']=='blocked_before_numerical_execution' and not plan['execution_authorized']
    assert not budget['approved'] and budget['approval_evidence'] is None and plan['blockers']==['no_approved_total_compute_budget']
    assert not any((out/name).exists() for name in ('records','events','workers','active.json','models'))
    assert sha(MODEL/'registry.json')==plan['registry_sha256']==pred['registry_sha256']==read(MODEL/'VALIDATION_DESIGN.json')['frozen_registry_sha256']
    assert sha(ROOT/'ml/probabilistic_selector.py')==plan['policy_code_sha256']==pred['policy_code_sha256']
    contract=read(CONTRACT/'plan.json');inputs=load_inputs(ROOT/contract['data_dir']);meta=inputs['metadata']
    for name,h in read(MODEL/'artifact_sha256.json').items():assert sha(MODEL/name)==h,name
    assert len(plan['jobs'])==len({j['key'] for j in plan['jobs']})==96
    assert Counter(j['profile'] for j in plan['jobs'])=={'R128':32,'R256':32,'MC1':32}
    assert Counter(j['equation_id'] for j in plan['jobs'])=={c['equation_id']:12 for c in plan['cases']}
    for j in plan['jobs']:
        m=meta[j['equation_id']];assert j['split']==m['split']=='validation' and j['expression_hash']==m['expression_hash']
        assert j['family_id']==m['family_id'] and not j['execution_authorized'] and j['target_x']==contract['target_x']
        cfg=j['config'];p=j['profile']
        if p=='MC1':
            expected=dict(contract['profiles'][p]['config'],seed=248800060100+j['episode'],max_path_steps=33*4096*j['expected_steps'],max_runtime_seconds=j['proposed_watchdog_seconds'])
            assert cfg==expected and j['expected_steps']==math.ceil(j['expected_T']/cfg['dt'])
            assert j['numerical_work']==cfg['max_path_steps']
        else:assert cfg==dict(contract['profiles'][p]['config'],T=j['expected_T']) and j['numerical_work']==cfg['n']*512
    assert math.isclose(sum(j['scenario_seconds'] for j in plan['jobs']),budget['total_scenario_seconds'],rel_tol=0,abs_tol=1e-9)
    assert budget['proposed_series_cap_seconds']==math.ceil(2*budget['total_scenario_seconds']+300)
    factor=budget['calibration_factor'];x0_sources=read(out/'budget_sources_sha256.json')
    assert len(x0_sources)==32
    for name,h in x0_sources.items():assert sha(ROOT/name)==h,name
    for c in plan['cases']:
        values=[read(ROOT/f"results/mc1000_x0_2026-10-25/attempts/{c['equation_id']}_seed{i}.json")['result']['runtime_seconds'] for i in range(4)]
        assert statistics.mean(values)==c['MC1_scenario_one_call_seconds']/(132*factor) or math.isclose(statistics.mean(values)*132*factor,c['MC1_scenario_one_call_seconds'],abs_tol=1e-12)
    selected_families={c['family_id'] for c in plan['cases']};assert len(selected_families)==8
    api=ProbabilisticSelector(MODEL/'registry.json');registry=api.registry
    assert all(not selected_families&set(h['metadata']['families']) for h in registry['heads'].values())
    index={str(pid):i for i,pid in enumerate(inputs['ids'])}
    estimates=estimates_from_heads(api.heads,inputs['X'][[index[c['equation_id']] for c in plan['cases']]],inputs['feature_names'])
    for row,est in zip(pred['rows'],estimates):
        assert row['estimates']==est and row['production_action']=='abstain'
        for q in (.5,.8,.95):
            assert row['diagnostic_choices'][str(q)]['point']==choose(est,q)
            assert row['diagnostic_choices'][str(q)]['model_spread']==choose(est,q,robust=True)
    before=read(out/'protected_before.json');after=snapshot(out)
    changed={name for name,h in before.items() if after.get(name)!=h};assert changed<= {'PROJECT_STATUS.md'},changed
    result=dict(passed=True,status='technical_preflight_passed_budget_not_approved',prepared_jobs=96,validation_equations=8,validation_families=8,
        fit_family_overlap=0,model_registry_sha256=plan['registry_sha256'],frozen_predictions_reproduced=True,
        budget_sources=32,scenario_seconds=budget['total_scenario_seconds'],proposed_cap_seconds=budget['proposed_series_cap_seconds'],budget_approved=False,
        solver_calls=0,MC_calls=0,ML_fits=0,benchmark1000_calls=0,
        previous_paths_checked=len(before),previous_paths_changed=sorted(changed),numerical_execution_artifacts_created=False)
    write(out/'independent_verification.json',result);print(result)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);verify(ap.parse_args().out.resolve())
