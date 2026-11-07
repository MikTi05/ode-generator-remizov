"""Frozen selection and analysis policy for the limited RK4/PI cost pilot."""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
DATA=Path('data/all_methods_v1_1000_2026-10-18')
SPLITS=Path('results/ml_selector_hypotheses_2026-10-31/splits.json')
OTHER_TEST=Path('results/ml_selector_final_2026-10-27/final_experiment/test_inputs/all_methods_equations_v1_metadata.jsonl')
METHODS=('rk4','product_integral')
NS=(32,64,128,256)
EPS=.01
REPEATS=5
FEATURES=('a_lower','a_upper','b_abs','c_R_abs','lambda','gap','a_span','b_span','c_R_span','g_abs')
POLICY=dict(minority_fraction=.20,minimum_pooled_saving=.10,minimum_mean_relative_saving=.10,
            practical_speed_margin=.10,max_relative_MAD=.20,minimum_stable_profile_fraction=.90,
            minimum_robust_each_method=12,minimum_total_solved=60)


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def digest_key(*parts):return hashlib.sha256(('cost60_v1|2488|'+'|'.join(map(str,parts))).encode()).hexdigest()


def atomic_json(path,value):
    """Exclusive atomic commit; never replace an existing record."""
    import os,tempfile
    path=Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.',suffix='.tmp',mode='w',delete=False) as stream:
        temporary=Path(stream.name)
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
        stream.flush();os.fsync(stream.fileno())
    try:os.link(temporary,path)
    finally:temporary.unlink(missing_ok=True)


def descriptor(meta):
    ranges=meta['validation']['sampled_ranges']
    return [meta['a_lower_cert'],meta['a_upper_cert'],meta['b_abs_cert'],meta['remizov_c_abs_cert'],
            meta['remizov_lambda'],meta['remizov_gap_cert'],
            *[ranges[c]['max']-ranges[c]['min'] for c in ('a','b','c_r')],meta['validation']['max_abs']['g']]


def select_metadata(metadata,splits,other_test):
    """One hashed representative per train family; three farthest points/cell."""
    held_ids=set().union(*(set(s['test']['ids']) for s in splits['scenarios'].values()))
    held_families={m['family_id'] for m in metadata if m['equation_id'] in held_ids}
    held_signatures={m['structure_signature'] for m in other_test}
    families=defaultdict(list)
    for m in metadata:
        if m['split']=='train' and m['family_id'] not in held_families and m['structure_signature'] not in held_signatures:
            families[m['family_id']].append(m)
    candidates=[min(group,key=lambda m:digest_key('representative',m['equation_id'])) for group in families.values()]
    cells=defaultdict(list)
    for m in candidates:cells[(m['complexity_level'],m['regime'])].append(m)
    require(len(cells)==20,'need all 20 input strata')
    chosen=[];selection=[];pool=[]
    for cell,items in sorted(cells.items()):
        items=sorted(items,key=lambda m:digest_key('family',m['family_id']))
        require(len(items)>=3,'not enough development families in '+str(cell))
        vectors={m['equation_id']:descriptor(m) for m in items}
        low=[min(v[j] for v in vectors.values()) for j in range(len(FEATURES))]
        high=[max(v[j] for v in vectors.values()) for j in range(len(FEATURES))]
        scaled={pid:[(v[j]-low[j])/(high[j]-low[j]) if high[j]!=low[j] else 0.
                     for j in range(len(FEATURES))] for pid,v in vectors.items()}
        picked=[items[0]]
        while len(picked)<3:
            def distance(m):
                return min(sum((a-b)**2 for a,b in zip(scaled[m['equation_id']],scaled[p['equation_id']])) for p in picked)
            picked.append(min((m for m in items if m not in picked),key=lambda m:(-distance(m),digest_key('family',m['family_id']))))
        for m in items:
            base={k:m[k] for k in ('equation_id','family_id','expression_hash','structure_signature','complexity_level','regime','split')}
            base.update(dict(zip(FEATURES,vectors[m['equation_id']])))
            pool.append(dict(base,selected=m in picked))
        for index,m in enumerate(picked):
            chosen.append(m)
            selection.append(dict(next(r for r in pool if r['equation_id']==m['equation_id']),selection_order_in_cell=index))
    order=sorted(range(len(chosen)),key=lambda i:digest_key('equation_order',chosen[i]['equation_id']))
    chosen=[chosen[i] for i in order];selection=[selection[i] for i in order]
    require(len(chosen)==len({m['family_id'] for m in chosen})==len({m['structure_signature'] for m in chosen})==60,'selection must have 60 distinct families/topologies')
    return chosen,selection,pool,dict(canonical_test_families=sorted(held_families),other_test_topologies=len(held_signatures),
                                     candidate_families=len(candidates),historical_test_selected=0)


def configuration(method,n):return f'{method}_N{n}'


def summarize_profile(records):
    times=[r['solver_seconds'] for r in records]
    require(len(records)==REPEATS and all(math.isfinite(t) and t>0 for t in times),'five positive measured times required')
    median=statistics.median(times);mad=statistics.median(abs(t-median) for t in times)
    first=records[0]
    trajectory=[r['result'] for r in records]
    identical=all(r==trajectory[0] for r in trajectory)
    errors=[r['E33'] for r in records]
    available=all(r['usable'] for r in records)
    return dict(equation_id=first['equation_id'],family_id=first['family_id'],method=first['method'],N=first['N'],
        configuration=configuration(first['method'],first['N']),usable=available,repeat_identical=identical,
        E33=errors[0] if available and identical else None,
        accurate=available and identical and all(e<=EPS for e in errors),
        solver_median_seconds=median,solver_min_seconds=min(times),solver_max_seconds=max(times),
        solver_MAD_seconds=mad,relative_MAD=mad/median,timing_stable=mad/median<=POLICY['max_relative_MAD'],
        failure_reason='' if available else '|'.join(sorted({r['status'] for r in records if not r['usable']})),
        times=times)


def choose(profiles):
    valid=[p for p in profiles if p['accurate']]
    return min(valid,key=lambda p:(p['solver_median_seconds'],METHODS.index(p['method']),p['N'])) if valid else None


def robust_faster(fast,slow):
    return bool(fast and slow and fast['timing_stable'] and slow['timing_stable']
        and fast['solver_median_seconds']<=(1-POLICY['practical_speed_margin'])*slow['solver_median_seconds']
        and fast['solver_max_seconds']<slow['solver_min_seconds'])


def saving(baseline,selected):
    require(len(baseline)==len(selected) and bool(baseline),'paired costs required')
    reductions=[1-b/a for a,b in zip(baseline,selected)]
    return dict(n=len(baseline),baseline_mean_seconds=statistics.mean(baseline),selected_mean_seconds=statistics.mean(selected),
                mean_absolute_saving_seconds=statistics.mean(a-b for a,b in zip(baseline,selected)),
                pooled_saving_fraction=1-sum(selected)/sum(baseline),mean_relative_saving_fraction=statistics.mean(reductions),
                median_relative_saving_fraction=statistics.median(reductions),total_baseline_seconds=sum(baseline),
                total_selected_seconds=sum(selected))
