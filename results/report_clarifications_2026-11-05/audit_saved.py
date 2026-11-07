"""Independent arithmetic on saved records. No solvers, exact evaluation or ML."""
from pathlib import Path
from collections import Counter,defaultdict
import csv
import hashlib
import json
import math
import statistics

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sources={}
checks=[]
def path(name):
    p=ROOT/name;sources[name]=hashlib.sha256(p.read_bytes()).hexdigest();return p
def read(name):return json.loads(path(name).read_text())
def rows(name):return list(csv.DictReader(path(name).open()))
def check(name,condition):
    checks.append(dict(check=name,passed=bool(condition)))
    assert condition,name
meta=[json.loads(s) for s in path('data/all_methods_v1_1000_2026-10-18/all_methods_equations_v1_metadata.jsonl').read_text().splitlines()]
by={m['equation_id']:m for m in meta}
grid=rows('results/saved_grid33_1000_2026-11-04/grid33_all1000.csv')
splits=read('results/ml_selector_hypotheses_2026-10-31/splits.json')['scenarios']
split_counts=Counter(m['split'] for m in meta)
check('raw dataset 700/150/150',split_counts==dict(train=700,validation=150,test=150))
check('ML split dimensions',[[len(splits[s][k]['ids']) for k in ('train','validation','test')] for s in ('old','new')]==[[697,150,150],[747,100,150]])
test_ids=set(splits['new']['test']['ids']);check('same historical Test',test_ids==set(splits['old']['test']['ids']))
test_families={by[i]['family_id'] for i in test_ids}
other=[json.loads(s) for s in path('results/ml_selector_final_2026-10-27/final_experiment/test_inputs/all_methods_equations_v1_metadata.jsonl').read_text().splitlines()]
other_signatures={m['structure_signature'] for m in other}
def eligible(m):return m['family_id'] not in test_families and m['structure_signature'] not in other_signatures
train=[m for m in meta if m['split']=='train' and eligible(m)]
non_test=[r for r in grid if by[r['equation_id']]['split']!='test' and eligible(by[r['equation_id']])]
check('cost pool 700; boundary pool 850',len(train)==700 and len(non_test)==850)
missing=[r['equation_id'] for r in grid if not r['R_E33']]
expected_missing=['AMV1_F0134_S03','AMV1_F0136_S01','AMV1_F0139_S00']
check('missing IDs and original Train',missing==expected_missing and all(by[i]['split']=='train' for i in missing))
bad=[r for r in grid if r['R_E33'] and float(r['R_E33'])>.05]
bad_test=[r['equation_id'] for r in bad if r['equation_id'] in test_ids]
check('27 exceedances; 5 Test; 22 outside Test',len(bad)==27 and len(bad_test)==5 and sum(bool(r['R_E33']) for r in non_test)==847)
boundary=read('results/remizov_boundary_dev_2026-11-05/plan.json')['selection_audit']
check('boundary audit matches independent count',boundary['development_tasks']==850 and boundary['case_count']==22 and boundary['missing_R_in_development']==missing)

# Independently reproduce the deterministic *input-only* choice, not any solution.
key=lambda *parts:hashlib.sha256(('cost60_v1|2488|'+'|'.join(map(str,parts))).encode()).hexdigest()
families=defaultdict(list)
for m in train:families[m['family_id']].append(m)
representatives=[min(g,key=lambda m:key('representative',m['equation_id'])) for g in families.values()]
def descriptor(m):
    v=m['validation'];r=v['sampled_ranges']
    return [m['a_lower_cert'],m['a_upper_cert'],m['b_abs_cert'],m['remizov_c_abs_cert'],m['remizov_lambda'],m['remizov_gap_cert'],
            *[r[k]['max']-r[k]['min'] for k in ('a','b','c_r')],v['max_abs']['g']]
cells=defaultdict(list)
for m in representatives:cells[m['complexity_level'],m['regime']].append(m)
selected=[]
for cell,group in sorted(cells.items()):
    group=sorted(group,key=lambda m:key('family',m['family_id']))
    vectors={m['equation_id']:descriptor(m) for m in group}
    low=[min(v[j] for v in vectors.values()) for j in range(10)]
    high=[max(v[j] for v in vectors.values()) for j in range(10)]
    scaled={i:[(v[j]-low[j])/(high[j]-low[j]) if high[j]!=low[j] else 0 for j in range(10)] for i,v in vectors.items()}
    chosen=[group[0]]
    while len(chosen)<3:
        def distance(m):return min(sum((a-b)**2 for a,b in zip(scaled[m['equation_id']],scaled[p['equation_id']])) for p in chosen)
        chosen.append(min((m for m in group if m not in chosen),key=lambda m:(-distance(m),key('family',m['family_id']))))
    selected+=chosen
selected=sorted(selected,key=lambda m:key('equation_order',m['equation_id']))
saved_selection=rows('results/cost60_rk4_pi_2026-11-04_v2/selection.csv')
check('all 60 selected IDs and order reproduced', [m['equation_id'] for m in selected]==[m['equation_id'] for m in saved_selection])
check('60 unique families/topologies and 20 cells',len({m['family_id'] for m in selected})==len({m['structure_signature'] for m in selected})==60 and len(cells)==20)

profiles=rows('results/cost60_rk4_pi_2026-11-04_v2/profiles.csv')
winners=rows('results/cost60_rk4_pi_2026-11-04_v2/winners.csv')
stable=0
by_eq=defaultdict(list)
for r in profiles:
    times=json.loads(r['times']);median=statistics.median(times);mad=statistics.median(abs(t-median) for t in times)
    check('repeat statistics '+r['equation_id']+'/'+r['configuration'],len(times)==5 and median==float(r['solver_median_seconds']) and math.isclose(mad/median,float(r['relative_MAD']),abs_tol=1e-15))
    stable+=mad/median<=.2;by_eq[r['equation_id']].append(r)
close=[];accuracy_only=0;robust_time=0
for w in winners:
    own=[r for r in by_eq[w['equation_id']] if r['accurate']=='True']
    rk=min((r for r in own if r['method']=='rk4'),key=lambda r:float(r['solver_median_seconds']))
    pi=[r for r in own if r['method']=='product_integral']
    check('raw winner '+w['equation_id'],min(own,key=lambda r:float(r['solver_median_seconds']))['method']=='rk4')
    if not pi:
        accuracy_only+=1;check('missing rival marked robust '+w['equation_id'],w['RK4_robust_win']=='True');continue
    pi=min(pi,key=lambda r:float(r['solver_median_seconds']))
    gain=1-float(rk['solver_median_seconds'])/float(pi['solver_median_seconds'])
    separated=float(rk['solver_max_seconds'])<float(pi['solver_min_seconds'])
    robust=gain>=.1 and separated and float(rk['relative_MAD'])<=.2 and float(pi['relative_MAD'])<=.2
    check('independent robust comparison '+w['equation_id'],robust==(w['RK4_robust_win']=='True'))
    if robust:robust_time+=1
    else:close.append(dict(equation_id=w['equation_id'],RK4_N=int(rk['N']),PI_N=int(pi['N']),
                          RK4_median_ms=1000*float(rk['solver_median_seconds']),PI_median_ms=1000*float(pi['solver_median_seconds']),
                          RK4_saving_percent=100*gain,intervals_separated=separated,
                          RK4_min_ms=1000*float(rk['solver_min_seconds']),RK4_max_ms=1000*float(rk['solver_max_seconds']),
                          PI_min_ms=1000*float(pi['solver_min_seconds']),PI_max_ms=1000*float(pi['solver_max_seconds'])))
check('stable 480; robust bookkeeping 53+3; 4 close',stable==480 and accuracy_only==53 and robust_time==3 and len(close)==4)
with (HERE/'close_cases.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(close[0]));writer.writeheader();writer.writerows(close)

available=[r for r in grid if r['R_E33']]
largest=sorted(available,key=lambda r:-float(r['R_E33']))[:2]
share=100*sum(float(r['R_E33']) for r in largest)/sum(float(r['R_E33']) for r in available)
check('largest two 98.842625 percent',math.isclose(share,98.84262516870092))
full_ids=['AMV1_'+v.replace('/','_') for v in ('F0141/S00','F0184/S01','F0186/S01','F0186/S02','F0189/S02','F0189/S03')]
check('six full equation IDs exist',all(i in by for i in full_ids))
registry=read('results/ml_three_method_selector_2026-11-03/selected_model/registry.json')
import numpy as np
with np.load(path('results/ml_three_method_selector_2026-11-03/features.npz'),allow_pickle=False) as z:
    check('exact 258 saved feature names',len(registry['feature_names'])==258 and registry['feature_names']==z['augmented_names'].tolist())
pred=rows('results/ml_three_method_selector_2026-11-03/test_predictions.csv')
pred=[r for r in pred if r['candidate_id']==registry['candidate']['id'] and r['policy']=='validation_optimal']
check('saved three probabilities sum to one',len(pred)==150 and all(len(p:=json.loads(r['probabilities']))==3 and all(0<=v<=1 for v in p) and abs(sum(p)-1)<1e-14 for r in pred))
importance=rows('results/ml_three_method_selector_2026-11-03/feature_importance.csv')
top=sorted((r for r in importance if r['candidate_id']==registry['candidate']['id']),key=lambda r:-float(r['importance']))[:5]
check('correct top features',{'a_total_variation','pre_diffusive_length'}<={r['feature'] for r in top})
old_train=set(splits['old']['train']['ids']);new_train=set(splits['new']['train']['ids'])
facts=dict(status='passed',checks=len(checks),raw_split_counts=dict(split_counts),cost_candidates=len(train),boundary_pool=len(non_test),
    boundary_available=sum(bool(r['R_E33']) for r in non_test),missing_remizov=missing,test_exceedances=bad_test,
    original_exceedance_counts=dict(Counter(by[r['equation_id']]['split'] for r in bad)),
    top_two=[dict(equation_id=r['equation_id'],E33=float(r['R_E33'])) for r in largest],top_two_share_percent=share,
    stable_timings=stable,accuracy_only_wins=accuracy_only,time_margin_wins=robust_time,close_cases=close,
    old_train_to_new_validation=len(old_train-new_train),old_validation_to_new_train=len(new_train-old_train),
    three_method_features=len(registry['feature_names']),biases=registry['biases'],feature_importance_top5=top,
    new_solver_calls=0,new_MC_calls=0,new_model_fits=0,new_model_predictions=0,all_checks=checks)
(HERE/'facts.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2)+'\n')
(HERE/'source_hashes.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in facts.items() if k not in ['all_checks','feature_importance_top5']},ensure_ascii=False))
