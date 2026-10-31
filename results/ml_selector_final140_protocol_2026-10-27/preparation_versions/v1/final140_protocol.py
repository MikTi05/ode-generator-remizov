"""Prepare/check the chosen fixed final140 protocol. Never solve or fit."""
from __future__ import annotations
import argparse,copy,csv,hashlib,json,math,shutil,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ml.revision_evidence import read,sha,verify_manifest
from diagnostics.prepare_development8 import write_json,utc,snapshot

RELEASE=ROOT/'results/ml_selector_final_2026-10-27'
SOURCE=RELEASE/'final_experiment'
REVIEW=ROOT/'results/ml_selector_budget_review_2026-10-27'
SEED=248800100100
ORDERS=(('R128','R256','MC1'),('R256','R128','MC1'),('MC1','R128','R256'),
        ('MC1','R256','R128'),('R128','MC1','R256'),('R256','MC1','R128'))


def schedule(cases,jobs,MC_ids):
    """Only measurement order changes; hash neither predictions nor observed costs."""
    def order(c):return hashlib.sha256(('final140_order_v1|'+c['identity']).encode()).hexdigest()
    mc=sorted((c for c in cases if c['equation_id'] in MC_ids),key=order)
    other=sorted((c for c in cases if c['equation_id'] not in MC_ids),key=order)
    routes={}
    for i,c in enumerate(mc):routes[c['equation_id']]=ORDERS[i%6] if i<18 else ORDERS[0 if i==18 else 3]
    for i,c in enumerate(other):routes[c['equation_id']]=('R128','R256') if i%2==0 else ('R256','R128')
    lookup={(j['equation_id'],j['profile']):j for j in jobs};result=[];ledger=[]
    for c in sorted(cases,key=order):
        pid=c['equation_id'];profiles=routes[pid]
        for p in profiles:result.append(copy.deepcopy(lookup[pid,p]))
        ledger.append(dict(equation_id=pid,identity=c['identity'],profiles=list(profiles),order_sha256=order(c)))
    return result,ledger


def validate_scope(plan):
    assert plan['schema']=='final_fixed140_v1' and plan['execution_authorized'] is False
    assert plan['resources']['series_seconds']==7464 and plan['resources']['compute_workers']==1 and plan['resources']['rss_mib']==512
    assert len(plan['cases'])==len({c['identity'] for c in plan['cases']})==len({c['structure_signature'] for c in plan['cases']})==60
    assert len(plan['MC_equation_ids'])==len(set(plan['MC_equation_ids']))==20
    assert plan['MC_seed']==SEED and plan['model_choices']=={'R128':21,'R256':39}
    assert Counter((c['complexity_level'],c['regime']) for c in plan['cases'])=={(l,r):3 for l in ('L1','L2','L3','L4') for r in ('constant','slow_varying','moderate_varying','oscillatory','high_contrast')}
    selected=[c for c in plan['cases'] if c['equation_id'] in plan['MC_equation_ids']]
    assert len({(c['complexity_level'],c['regime']) for c in selected})==20
    jobs=plan['jobs'];assert len(jobs)==len({j['key'] for j in jobs})==140
    assert Counter(j['profile'] for j in jobs)=={'R128':60,'R256':60,'MC1':20}
    source=read(ROOT/plan['source_plan']);assert sha(ROOT/plan['source_plan'])==plan['source_plan_sha256']
    options=read(ROOT/plan['source_options'])['options'];assert plan['MC_equation_ids']==options[0]['equation_ids']
    assert {j['key'] for j in jobs}==set(options[2]['referenced_primary_job_keys'])
    assert plan['cases']==source['cases'] and all(j in source['jobs'] and j['episode']==0 and j['split']=='test' for j in jobs)
    expected,ledger=schedule(plan['cases'],jobs,set(plan['MC_equation_ids']));assert expected==jobs and ledger==plan['measurement_order']
    assert sum(r['profiles'].index('R128')<r['profiles'].index('R256') for r in ledger)==30
    assert sum(r['profiles'].index('R128')<r['profiles'].index('R256') for r in ledger if r['equation_id'] in plan['MC_equation_ids'])==10


def prepare(out):
    assert not out.exists();verify_manifest(RELEASE);verify_manifest(REVIEW)
    source=read(SOURCE/'plan.json');options=read(REVIEW/'options.json')['options'];option=options[2]
    assert option['maximum_calls']==140 and option['proposed_cap_seconds']==7464
    assert not list((SOURCE/'records').glob('*.json')) and not list((SOURCE/'events').glob('*.request.json'))
    out.mkdir(parents=True,exist_ok=False);shutil.copy2(ROOT/'PROJECT_STATUS.md',out/'PROJECT_STATUS.before.md');write_json(out/'protected_before.json',snapshot(out),exclusive=True)
    for n in ('records','events','workers','sessions','sources','reconciliations'):(out/n).mkdir()
    shutil.copy2(SOURCE/'frozen_predictions.json',out/'frozen_predictions.json')
    shutil.copy2(SOURCE/'independence.json',out/'independence.json')
    MC_ids=options[0]['equation_ids'];jobs=[j for j in source['jobs'] if j['key'] in set(option['referenced_primary_job_keys'])]
    jobs,order=schedule(source['cases'],jobs,set(MC_ids))
    spec=dict(schema='final_fixed140_analysis_v1',fixed_n=60,primary_profiles=['R128','R256'],secondary_MC_families=20,
        model_policy_and_inputs_unchanged=True,primary_episode=0,repetitions=1,
        primary=['accurate deliveries/60','finite E_grid>.01 violations','refusals and execution failures','sum realized full warm method wall + frozen measured inference proxy','cost per accurate including inaccurate calls'],
        comparison='same saved R observation reused for selector and constant_R128/R256; no extra selected-profile solve',
        cost_estimand='one realized full warm call, not median4; no stable per-equation10percent winner claim',
        uncertainty=dict(main='10000 paired family bootstrap within20cells with3families each; descriptive, not a noninferiority certificate',
            conservative='bounded paired/individual-loss references under independent family sampling; zero empirical delta is not proof of equality',
            MC='one family per20cell: no within-cell bootstrap; conservative bounded mean references only; no per-family reliability or seed variability claim',
            mean_forecasts='cost head predicts typical training time; deviation from one new call is a noisy descriptive check'),
        MC_scope='secondary counterfactual20 only; MC for40 not_measured/unknown, never none; no60-family full3-profile oracle or MC-choice-quality claim',
        stops=dict(efficacy_or_futility=False,resources=['7464s original wall deadline','RSS>512MiB','thermal>=2 or unavailable','guardian/lease/telemetry failure'],
            critical=['hash/identity/config/work/seed mismatch','invalid numerical record','worker exception'],operator_stop=True,
            negative_or_null_quality_is_valid_result=True,no_model_tuning_or_new_seeds=True),
        resume='completed immutable; requested-but-unrecorded reserved attempt reconciled as interrupted_unknown after cleanup; never rerun; only not-yet-requested jobs may start; original epoch+monotonic deadlines survive',
        censoring='unknown numerical accuracy never fabricated; unattempted not counted as failures; incomplete coverage and missing/censored time explicitly reported',
        completion='140 requested slots accounted and validated; primary completeness additionally requires120 evaluable R outcomes; do not require ML/MC superiority; production stays blocked',
        execution_authorized=False)
    write_json(out/'analysis_spec.json',spec,exclusive=True)
    budget=dict(selected_for_protocol=True,execution_authorized=False,series_seconds=7464,maximum_calls=140,scenario_seconds=option['scenario_seconds'],
        components={p:sum(c[p+'_scenario_seconds'] for c in source['cases']) for p in ('R128','R256')},
        MC_scenario_seconds=sum(c['MC_scenario_seconds'] for c in source['cases'] if c['equation_id'] in MC_ids),
        formula='ceil(2*3581.8146079732246+300)=7464; no reset/extension',not_a_proven_runtime_bound=True)
    write_json(out/'budget_proposal.json',budget,exclusive=True)
    files=['diagnostics/final140_protocol.py','diagnostics/run_final140.py','diagnostics/evaluate_final140.py',
        'diagnostics/confirm_f0047_timing.py','diagnostics/remizov_r0.py','diagnostics/mc_pilot_p2.py','diagnostics/chernov_batch_control.py',
        'diagnostics/mc_production_x0.py','diagnostics/joint_benchmark_pilot.py','diagnostics/run_chernov_batch.py']
    pins=dict(source['protected_sha256']);pins.update({n:sha(ROOT/n) for n in files})
    for name in ('analysis_spec.json','budget_proposal.json','frozen_predictions.json','independence.json'):
        pins[str((out/name).relative_to(ROOT))]=sha(out/name)
    pins[str((SOURCE/'plan.json').relative_to(ROOT))]=sha(SOURCE/'plan.json');pins[str((REVIEW/'options.json').relative_to(ROOT))]=sha(REVIEW/'options.json')
    plan=dict(schema='final_fixed140_v1',created_utc=utc(),execution_authorized=False,user_instruction='variant C selected; protocol budget7464s; execution explicitly prohibited in preparation turn',
        source_plan=str((SOURCE/'plan.json').relative_to(ROOT)),source_plan_sha256=sha(SOURCE/'plan.json'),source_options=str((REVIEW/'options.json').relative_to(ROOT)),
        data_dir=source['data_dir'],target_x=source['target_x'],cases=source['cases'],jobs=jobs,MC_equation_ids=MC_ids,MC_seed=SEED,
        model_choices=read(out/'frozen_predictions.json')['choices'],measurement_order=order,protected_sha256=pins,
        resources=dict(source['resources'],series_seconds=7464),primary='selector vs constant_R128/R256 on60',secondary='MC counterfactual on fixed20 only',
        repetitions=1,maximum_calls=140,production_blocked=True,no_statistical_interim_stopping=True)
    validate_scope(plan);write_json(out/'plan.json',plan,exclusive=True)
    with (out/'manifest.csv').open('x',newline='') as f:
        fields=['position','key','identity','equation_id','family_id','profile','episode','seed','T','N','numerical_work','watchdog_seconds']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for i,j in enumerate(jobs):w.writerow(dict(position=i,key=j['key'],identity=j['identity'],equation_id=j['equation_id'],family_id=j['family_id'],profile=j['profile'],episode=0,seed=j['config'].get('seed'),T=j['expected_T'],N=j['expected_steps'],numerical_work=j['numerical_work'],watchdog_seconds=j['proposed_watchdog_seconds']))
    with (out/'MC_subset20.csv').open('x',newline='') as f:
        fields=['equation_id','identity','structure_signature','complexity_level','regime','selection_sha256'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for c in source['cases']:
            if c['equation_id'] in MC_ids:w.writerow({n:c[n] for n in fields})
    write_json(out/'approval_request.json',dict(approved=False,execution_authorized=False,plan_sha256=sha(out/'plan.json'),maximum_calls=140,series_seconds=7464,rss_mib=512,workers=1,threads=1,
        budget_already_fixed_for_protocol=True,separate_start_instruction_required=True),exclusive=True)
    from diagnostics.run_final140 import preflight
    check=preflight(out);write_json(out/'preflight.json',check,exclusive=True);print(check)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);prepare(ap.parse_args().out.resolve())
