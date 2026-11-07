#!/usr/bin/env python3
"""Independent saved-answer verifier for boundary development intervention."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def require(value,message):
    if not value:raise ValueError(message)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def table(path):
    with path.open(newline='',encoding='utf-8') as stream:return list(csv.DictReader(stream))
def compare_cell(got,value,label):
    if value is None:require(got=='',label)
    elif isinstance(value,bool):require(got==str(value),label)
    elif isinstance(value,(int,float)):require(float(got)==value,label)
    else:require(got==value,label)


def verify(out):
    p=read(out/'plan.json');s=read(out/'summary.json');session=read(out/'session.json');plan_sha=sha(out/'plan.json')
    require(all(sha(ROOT/name)==digest for name,digest in p['source_sha256'].items()),'pinned source/data changed')
    require(sha(out/'selection.csv')==p['selection_sha256'],'selection hash')
    require(p['created_utc']<=read(out/'started.json')['utc'] and read(out/'started.json')['plan_sha256']==plan_sha,'plan must precede run')
    config=p['baseline_config'];zero=p['zero_config']
    require([k for k in config if config[k]!=zero[k]]==['boundary'] and config['boundary']=='taper' and zero['boundary']=='zero','boundary-only intervention')
    require(config['interp']=='cubic' and p['epsilon']==.05 and p['catastrophe_threshold']==1,'fixed comparison')
    metadata={m['equation_id']:m for m in map(json.loads,(ROOT/'data/all_methods_v1_1000_2026-10-18/all_methods_equations_v1_metadata.jsonl').read_text().splitlines())}
    splits=read(ROOT/'results/ml_selector_hypotheses_2026-10-31/splits.json')
    held=set().union(*(set(x['test']['ids']) for x in splits['scenarios'].values()))
    families={metadata[pid]['family_id'] for pid in held}
    other={m['structure_signature'] for m in map(json.loads,(ROOT/'results/ml_selector_final_2026-10-27/final_experiment/test_inputs/all_methods_equations_v1_metadata.jsonl').read_text().splitlines())}
    baseline_rows=table(ROOT/'results/saved_grid33_1000_2026-11-04/grid33_all1000.csv')
    dev=[r for r in baseline_rows if metadata[r['equation_id']]['split']!='test' and r['family_id'] not in families and metadata[r['equation_id']]['structure_signature'] not in other]
    expected_bad={r['equation_id'] for r in dev if r['R_E33'] and float(r['R_E33'])>.05}
    tasks=p['tasks'];by_id={r['equation_id']:r for r in tasks};published=table(out/'before_after.csv')
    require(len(tasks)==len(by_id)==len(published)==52,'fixed sample cardinality')
    require({r['equation_id'] for r in tasks if r['category']=='case'}==expected_bad and len(expected_bad)==22,'all dev exceedances')
    control=[r for r in tasks if r['category']=='control'];casefamilies={r['family_id'] for r in tasks if r['category']=='case'}
    require(len(control)==len({r['family_id'] for r in control})==30 and not casefamilies&{r['family_id'] for r in control},'distinct control families')
    for r in tasks:
        m=metadata[r['equation_id']]
        require(m['split']!='test' and m['family_id'] not in families and m['structure_signature'] not in other,'Test contamination')
        require(all(m[k]==r[k] for k in ('family_id','expression_hash','structure_signature','complexity_level','regime','split')),'selection identity')
    require(Counter(r['complexity_level']+'/'+r['regime'] for r in control)==p['selection_audit']['control_family_quotas'],'control strata')
    csv_map={r['equation_id']:r for r in published};raw_paths=set();derived=[];records=[]
    for pid,t in by_id.items():
        before=read(ROOT/t['before_source']);after=read(out/'records'/(pid+'.json'));records.append(after)
        require(sha(ROOT/t['before_source'])==t['before_sha256']==after['before_sha256'],'baseline checksum')
        require(before['metadata']==metadata[pid] and before['config']==config,'baseline identity/config')
        require(after['plan_sha256']==plan_sha and after['config']==zero and after['expression_hash']==metadata[pid]['expression_hash'],'zero config/identity')
        require(after['y_exact']==before['y_exact'] and after['fingerprints']==t['fingerprints'],'unchanged exact/numerical inputs')
        require(after['reused']==bool(t['cached_zero_source']),'reuse policy')
        q0=before['result'];q=after['result'];usable=bool(q and q['success'] and q['failure_reason']=='')
        require(q0['success'] and q0['x']==before['target_x']==[-1+i/16 for i in range(33)],'baseline success/target')
        e0=max(abs(a-b) for a,b in zip(q0['f'],before['y_exact']));require(e0==t['E_before']==before['max_abs_error'],'baseline E33')
        e1=None
        if usable:
            require(q['scheme']=='solve' and q['x']==q0['x'] and len(q['f'])==33 and all(math.isfinite(v) for v in q['f']),'zero target/finite')
            for field in ('T','n','n_steps','time_nodes','actual_dx','work_grid_points','work_domain','n_operator_applications','n_spline_builds','outside_query_values'):
                require(q[field]==q0[field],'geometry/work changed: '+field)
            require(q['n_operator_applications']==65536 and q['n_spline_builds']==65537 and q['work_grid_points']==2081,'fixed workload')
            e1=max(abs(a-b) for a,b in zip(q['f'],before['y_exact']));require(e1==after['max_abs_error'] and after['accuracy_met']==(e1<=.05),'zero E33')
        else:require(after['max_abs_error'] is None and after['accuracy_met'] is None,'failed zero fabricated error')
        if t['cached_zero_source']:
            old=read(ROOT/t['cached_zero_source'])
            require(old['result']==q and old['config']==zero and old['production_result_sha256']==t['before_sha256'],'cached answer unchanged')
        delta=e1-e0 if usable else None;tol=1e-10*max(1,e0)
        row={k:t[k] for k in ('equation_id','category','E_before','family_id','expression_hash','structure_signature','complexity_level','regime','split')}
        row.update(E_after=e1,delta_error=delta,ratio_after_before=e1/e0 if usable and e0 else None,
            pass_before=e0<=.05,pass_after=e1<=.05 if usable else None,usable_zero=usable,
            improved=bool(usable and delta< -tol),equal_within_tolerance=bool(usable and abs(delta)<=tol),
            increased_unrounded=bool(usable and delta>0),material_regression=bool(usable and delta>tol),comparison_tolerance=tol,
            pass_lost=bool(usable and e0<=.05 and e1>.05),pass_gained=bool(usable and e0>.05 and e1<=.05),
            catastrophe_before=e0>1,catastrophe_after=e1>1 if usable else None,new_catastrophe=bool(usable and e0<=1 and e1>1),
            before_source=after['before_source'],zero_source=after['zero_source'],zero_reused=after['reused'],
            boundary_max_before=q0['boundary_state_max_abs'],boundary_max_after=q.get('boundary_state_max_abs') if q else None,
            T=q0['T'],failure_reason=q.get('failure_reason',after.get('exception','missing result')) if not usable and q else (after.get('exception') or 'missing result') if not usable else '')
        require(set(row)==set(csv_map[pid]),'CSV schema')
        for k,value in row.items():compare_cell(csv_map[pid][k],value,pid+'/'+k)
        derived.append(row);raw_paths.add(out/'records'/(pid+'.json'))
    require(table(out/'regressions.csv')==[r for r in published if r['increased_unrounded']=='True'],'all increases listed')
    require(table(out/'material_regressions.csv')==[r for r in published if r['material_regression']=='True'],'material regressions listed')
    require(table(out/'unresolved_cases.csv')==[r for r in published if r['pass_after']!='True'],'unresolved cases listed')
    for label in ('case','control','all'):
        rr=derived if label=='all' else [r for r in derived if r['category']==label];g=s['groups'][label]
        expected=dict(n=len(rr),before_pass=sum(r['pass_before'] for r in rr),after_pass=sum(r['pass_after'] is True for r in rr),
            improved=sum(r['improved'] for r in rr),equal=sum(r['equal_within_tolerance'] for r in rr),
            increased_unrounded=sum(r['increased_unrounded'] for r in rr),material_regressions=sum(r['material_regression'] for r in rr),
            pass_losses=sum(r['pass_lost'] for r in rr),pass_gains=sum(r['pass_gained'] for r in rr),
            catastrophes_before=sum(r['catastrophe_before'] for r in rr),catastrophes_after=sum(r['catastrophe_after'] is True for r in rr),
            maximum_before=max(r['E_before'] for r in rr),maximum_after=max((r['E_after'] for r in rr if r['E_after'] is not None),default=None))
        require(g==expected,'group summary '+label)
    gates=dict(all_zero_results_usable=all(r['usable_zero'] for r in derived),no_material_regressions=not any(r['material_regression'] for r in derived),
        no_pass_losses=not any(r['pass_lost'] for r in derived),no_new_catastrophes=not any(r['new_catastrophe'] for r in derived),
        former_catastrophes_removed=all(r['usable_zero'] and not r['catastrophe_after'] for r in derived if r['catastrophe_before']),
        pass_count_increases=sum(r['pass_after'] is True for r in derived)>sum(r['pass_before'] for r in derived),
        at_least_one_material_improvement=any(r['improved'] for r in derived))
    require(s['GO_gates']==gates and s['recommendation']==('GO' if all(gates.values()) else 'NO-GO'),'independent GO')
    journal=[json.loads(line) for line in (out/'call_journal.jsonl').read_text().splitlines()]
    calls=[j for j in journal if j['event']=='start']
    require(len(calls)==len({j['problem_id'] for j in calls})==p['new_zero_call_budget']==50,'no repeated/extra zero calls')
    require({j['problem_id'] for j in calls}=={t['equation_id'] for t in tasks if not t['cached_zero_source']},'call set')
    require(len([j for j in journal if j['event']=='saved'])==52 and session['complete'] and session['new_zero_calls']==50 and session['reused_zero']==2 and session['new_taper_calls']==0,'session budget')
    require(session['MC_calls']==session['ML_calls']==session['other_continuations_called']==0,'forbidden calls')
    report=(out/'REPORT.md').read_text()
    for r in derived:
        require(r['equation_id'] in report and f"{r['E_before']:.12g}" in report,'report baseline rounding')
        if r['E_after'] is not None:require(f"{r['E_after']:.12g}" in report and f"{r['delta_error']:.12g}" in report,'report zero rounding')
    require('экспериментальн' in report and s['recommendation'] in report,'report decision scope')
    artifacts=['plan.json','selection.csv','before_after.csv','regressions.csv','material_regressions.csv','unresolved_cases.csv','summary.json','REPORT.md','SPATIAL_ASSESSMENT.md']
    require(all(sha(ROOT/name)==digest for name,digest in p['source_sha256'].items()),'final source checksum')
    result=dict(passed=True,selected_tasks=52,cases=22,controls=30,development_only=True,historical_Test_overlap=0,
        all_before_after_cells_checked=len(published)*len(published[0]),baseline_errors_recomputed=52,zero_errors_recomputed=52,
        matching_fingerprints_and_geometry=52,reused_zero_answers_checked=2,new_zero_call_journal_checked=50,new_taper_calls=0,
        all_increases_and_material_regressions_checked=True,GO_independently_recomputed=True,
        recommendation=s['recommendation'],new_solver_calls_in_verifier=0,
        artifacts_sha256={name:sha(out/name) for name in artifacts},raw_zero_sha256={str(p.relative_to(out)):sha(p) for p in sorted(raw_paths)})
    require(not (out/'verification.json').exists(),'verification exists; use new replay directory')
    (out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print({k:v for k,v in result.items() if k not in ('artifacts_sha256','raw_zero_sha256')})
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    verify(parser.parse_args().out.resolve())
