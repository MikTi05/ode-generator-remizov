"""Frozen development selection and comparison for the cubic/zero experiment."""
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=Path('data/all_methods_v1_1000_2026-10-18')
BASE=Path('results/chernov_solve1000_2026-10-18/solutions')
GRID_RELEASE=Path('results/saved_grid33_1000_2026-11-04')
OLD_DIAG=Path('results/chernov_boundary_diagnosis_2026-10-24')
SPLITS=Path('results/ml_selector_hypotheses_2026-10-31/splits.json')
OTHER_TEST=Path('results/ml_selector_final_2026-10-27/final_experiment/test_inputs/all_methods_equations_v1_metadata.jsonl')
EPS=.05
CAT=1.
CONFIG=dict(T=None,boundary='taper',dx=.0125,interp='cubic',max_work_grid_points=100000,
            n=128,n_jobs=1,n_steps=32768,num_t=513,padding=12.,start_method='spawn',
            t_grading='quadratic',tail_eps=1e-8,taper_width=None)


def require(value,message):
    if not value:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def key(*parts):return hashlib.sha256(('boundary_dev_v1|2488|'+'|'.join(map(str,parts))).encode()).hexdigest()
def table(path):
    with Path(path).open(newline='',encoding='utf-8') as stream:return list(csv.DictReader(stream))
def csv_write(path,rows,fields=None):
    with Path(path).open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields or list(rows[0]));writer.writeheader();writer.writerows(rows)
def atomic_json(path,value):
    import os,tempfile
    path=Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.',suffix='.tmp',mode='w',delete=False) as stream:
        temporary=Path(stream.name)
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    try:os.link(temporary,path)
    finally:temporary.unlink(missing_ok=True)


def select(rows,metadata,splits,other_test):
    meta={m['equation_id']:m for m in metadata}
    test_ids=set().union(*(set(s['test']['ids']) for s in splits['scenarios'].values()))
    test_families={meta[pid]['family_id'] for pid in test_ids}
    other_signatures={m['structure_signature'] for m in other_test}
    dev=[r for r in rows if meta[r['equation_id']]['split']!='test' and r['family_id'] not in test_families
         and meta[r['equation_id']]['structure_signature'] not in other_signatures]
    bad=[r for r in dev if r['R_E33'] and float(r['R_E33'])>EPS]
    bad_families={r['family_id'] for r in bad}
    good=[r for r in dev if r['R_E33'] and float(r['R_E33'])<=EPS and r['family_id'] not in bad_families]
    cells=defaultdict(lambda:defaultdict(list))
    for r in good:
        m=meta[r['equation_id']];cells[m['complexity_level'],m['regime']][r['family_id']].append(r)
    levels=('L1','L2','L3','L4');regimes=('constant','slow_varying','moderate_varying','oscillatory','high_contrast')
    controls=[];quotas={}
    for j,regime in enumerate(regimes):
        for i,level in enumerate(levels):
            quota=1+int(i in (j%4,(j+1)%4));quotas[level+'/'+regime]=quota
            families=cells[level,regime]
            require(len(families)>=quota,'insufficient control families in '+level+'/'+regime)
            for family in sorted(families,key=lambda f:key('control-family',f))[:quota]:
                controls.append(min(families[family],key=lambda r:key('control-sample',r['equation_id'])))
    require(len(controls)==len({r['family_id'] for r in controls})==30,'30 distinct control families required')
    selected=[]
    for category,items in (('case',bad),('control',controls)):
        for r in sorted(items,key=lambda r:r['equation_id']):
            m=meta[r['equation_id']]
            selected.append(dict(equation_id=r['equation_id'],category=category,E_before=float(r['R_E33']),
                **{k:m[k] for k in ('family_id','expression_hash','structure_signature','complexity_level','regime','split')}))
    require(not {r['family_id'] for r in controls}&bad_families,'controls overlap case families')
    return selected,dict(development_tasks=len(dev),case_count=len(bad),control_count=len(controls),
        case_families=len(bad_families),control_family_quotas=quotas,
        missing_R_in_development=[r['equation_id'] for r in dev if not r['R_E33']],
        Test_overlap=0,selection_uses_only_before_error_and_metadata=True)


def max_error(values,exact):
    require(len(values)==len(exact)==33 and all(math.isfinite(v) for v in [*values,*exact]),'33 finite values required')
    return max(abs(v-e) for v,e in zip(values,exact))


def compare(selected,before,after):
    e0=max_error(before['result']['f'],before['y_exact'])
    result=after.get('result') or {}
    usable=result.get('success') is True and result.get('failure_reason')==''
    e1=max_error(result['f'],after['y_exact']) if usable else None
    tolerance=1e-10*max(1,e0)
    delta=e1-e0 if usable else None
    return dict(**selected,E_after=e1,delta_error=delta,ratio_after_before=e1/e0 if usable and e0 else None,
        pass_before=e0<=EPS,pass_after=e1<=EPS if usable else None,usable_zero=usable,
        improved=bool(usable and delta < -tolerance),equal_within_tolerance=bool(usable and abs(delta)<=tolerance),
        increased_unrounded=bool(usable and delta>0),material_regression=bool(usable and delta>tolerance),
        comparison_tolerance=tolerance,pass_lost=bool(usable and e0<=EPS and e1>EPS),
        pass_gained=bool(usable and e0>EPS and e1<=EPS),catastrophe_before=e0>CAT,
        catastrophe_after=e1>CAT if usable else None,new_catastrophe=bool(usable and e0<=CAT and e1>CAT),
        before_source=after['before_source'],zero_source=after['zero_source'],zero_reused=after['reused'],
        boundary_max_before=before['result']['boundary_state_max_abs'],
        boundary_max_after=result.get('boundary_state_max_abs'),T=before['result']['T'],
        failure_reason=result.get('failure_reason',after.get('exception','missing result')) if not usable else '')


def summary(rows):
    gates=dict(all_zero_results_usable=all(r['usable_zero'] for r in rows),
        no_material_regressions=not any(r['material_regression'] for r in rows),
        no_pass_losses=not any(r['pass_lost'] for r in rows),no_new_catastrophes=not any(r['new_catastrophe'] for r in rows),
        former_catastrophes_removed=all(r['usable_zero'] and not r['catastrophe_after'] for r in rows if r['catastrophe_before']),
        pass_count_increases=sum(r['pass_after'] is True for r in rows)>sum(r['pass_before'] for r in rows),
        at_least_one_material_improvement=any(r['improved'] for r in rows))
    groups={}
    for name in ('case','control','all'):
        group=rows if name=='all' else [r for r in rows if r['category']==name]
        groups[name]=dict(n=len(group),before_pass=sum(r['pass_before'] for r in group),
            after_pass=sum(r['pass_after'] is True for r in group),improved=sum(r['improved'] for r in group),
            equal=sum(r['equal_within_tolerance'] for r in group),increased_unrounded=sum(r['increased_unrounded'] for r in group),
            material_regressions=sum(r['material_regression'] for r in group),pass_losses=sum(r['pass_lost'] for r in group),
            pass_gains=sum(r['pass_gained'] for r in group),catastrophes_before=sum(r['catastrophe_before'] for r in group),
            catastrophes_after=sum(r['catastrophe_after'] is True for r in group),
            maximum_before=max((r['E_before'] for r in group),default=None),maximum_after=max((r['E_after'] for r in group if r['E_after'] is not None),default=None))
    return dict(epsilon=EPS,catastrophe_threshold=CAT,n=len(rows),groups=groups,GO_gates=gates,
        recommendation='GO' if all(gates.values()) else 'NO-GO',failed_gates=[k for k,v in gates.items() if not v],
        scope='new experimental cubic/zero profile on this development sample; not production promotion or theorem certification',
        material_regression_ids=[r['equation_id'] for r in rows if r['material_regression']],
        pass_loss_ids=[r['equation_id'] for r in rows if r['pass_lost']],
        new_catastrophe_ids=[r['equation_id'] for r in rows if r['new_catastrophe']],
        failed_zero_ids=[r['equation_id'] for r in rows if not r['usable_zero']],
        reused_zero=sum(r['zero_reused'] for r in rows),new_zero_calls=sum(not r['zero_reused'] for r in rows),
        new_taper_calls=0,ML_calls=0,MC_calls=0,other_profiles_called=0,production_profile_changed=False)
