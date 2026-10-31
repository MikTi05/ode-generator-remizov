"""Independent audit of the completed pilot. No numerical solvers invoked."""
from collections import Counter,defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    plan=read(OUT/'plan.json');summary=read(OUT/'summary.json');stats=read(OUT/'statistics.json')
    before=read(OUT/'protected_before.json')
    changed=[n for n,h in before.items() if not (ROOT/n).exists() or sha(ROOT/n)!=h]
    assert not changed,changed
    assert summary['status']=='completed' and len(plan['attempts'])==len(summary['runs'])==262
    assert sha(OUT/'plan.json')==summary['plan_sha256']
    assert plan['resources']['attempt_wall_seconds'] is None
    assert plan['resources']['phase_wall_seconds'] is None
    assert plan['resources']['total_wall_seconds'] is None
    assert plan['resources']['workers']==plan['resources']['blas_threads']==1
    sources_changed=[n for n,h in plan['source_sha256'].items() if sha(ROOT/n)!=h]
    assert not sources_changed,sources_changed
    groups=defaultdict(list);recorded=Counter();W=0;S=0;last_finished=-math.inf;guard_count=0
    peak=0.;frozen_order=0;longer60=[];error_passes=Counter();work_prediction_pairs=0
    for spec,terminal in zip(plan['attempts'],summary['runs']):
        aid=spec['attempt_id'];trial=OUT/'trials'/aid
        assert terminal['attempt_id']==aid and terminal['status']=='completed'
        request,prediction,ready,score,final=[read(trial/f) for f in
             ('request.json','prediction.json','ready.json','score.json','worker.final.json')]
        assert request['config']==spec['config']==prediction['config']
        assert request['launch_monotonic']>=last_finished
        g=read(OUT/terminal['guardian'])
        assert g['finished'] and g['cleanup_success'] and g['worker_returncode']==0
        assert g['reason']=='worker_exit'
        last_finished=g['finished_monotonic'];guard_count+=1
        checksum=sha(trial/'prediction.json')
        assert checksum==ready['prediction_sha256']==score['prediction_sha256']
        assert ready['prediction_frozen_utc']<=score['scored_utc'];frozen_order+=1
        r=prediction['result'];f=r['f'];y=score['y_exact']
        assert r['success'] and r['x']==spec['target_x']
        assert len(f)==len(y)==len(spec['target_x']) and all(math.isfinite(x) for x in f+y)
        E=max(abs(a-b) for a,b in zip(f,y))
        assert E==score['max_abs_error']==terminal['max_abs_error']
        assert (E<=.01)==score['observed_accuracy_pass']==terminal['observed_accuracy_pass']
        assert r['T']==spec['config']['T']==plan['tasks'][spec['problem_id']]['T']
        assert prediction['common_tail_bound']<=1e-4 and r['total_error_certified'] is False
        key=(spec['mode'],spec['problem_id'],spec['method'])
        groups[key].append(dict(spec=spec,E=E,f=f,response=ready['worker_response_seconds']))
        recorded[(spec['mode'],spec['method'])]+=1
        error_passes[(spec['mode'],spec['method'])]+=int(E<=.01)
        peak=max(peak,final['peak_rss_mib'])
        if ready['worker_response_seconds']>60:longer60.append(aid)
        if spec['method']=='monte_carlo':
            M=spec['config']['M'];K=len(spec['target_x']);N=int(128*r['T'])
            assert r['dt_actual']==1/128 and r['n_steps']==N
            assert r['M']==M and r['completed_paths']==[M]*K
            assert r['n_path_steps']==r['n_normal_draws']==M*K*N
            assert r['seed_entropy']==spec['config']['seed']
            assert sum(r['coefficient_values'].values())==M*K*(4*N+2)
            assert r['tail_tolerance_met'] is True
            W+=r['n_path_steps']
        else:
            assert r['n_operator_applications']==65536 and r['n_spline_builds']==65537
            assert r['work_grid_points']==2081 and r['work_domain']==[-13.,13.]
            S+=r['n_operator_applications']
        work_prediction_pairs+=1
    assert W==4450484224 and S==5111808
    assert len(groups)==52 and guard_count==262
    for key,rr in groups.items():
        mode,pid,method=key
        assert len(rr)==(3 if method=='solve' else 8 if mode=='point' else 4)
        if method=='solve':assert all(v['f']==rr[0]['f'] for v in rr)
    profiles=list(csv.DictReader((OUT/'profiles.csv').open()))
    comparisons=list(csv.DictReader((OUT/'comparisons.csv').open()))
    assert len(profiles)==52 and len(comparisons)==26
    for p in profiles:
        rr=groups[(p['mode'],p['problem_id'],p['method'])]
        k=sum(x['E']<=.01 for x in rr);n=len(rr)
        assert int(p['accuracy_successes'])==k
        assert float(p['median_response_seconds'])==statistics.median(x['response'] for x in rr)
        assert float(p['median_error'])==statistics.median(x['E'] for x in rr)
        if p['method']=='monte_carlo':
            # Exact small-n binomial arithmetic, independently of scipy.stats.
            exact_p=math.fsum(math.comb(n,j)*.95**j*.05**(n-j) for j in range(k+1))
            assert math.isclose(float(p['below_target_p']),exact_p,rel_tol=1e-12,abs_tol=1e-15)
    assert all(p['training_eligible']=='False' for p in profiles)
    assert all(c['training_eligible']=='False' for c in comparisons)
    assert all(x['point_equals_grid_center_bitwise'] for x in stats['cross_mode_remizov'])
    active=read(OUT/'active.json');assert active['finished']
    broken=[]
    for doc in [OUT/'REPORT.md',ROOT/'PROJECT_STATUS.md']:
        for link in re.findall(r'\]\(([^)]+)\)',doc.read_text()):
            if '://' not in link and not link.startswith('#') and not (doc.parent/link.split('#')[0]).exists():
                broken.append(dict(document=str(doc.relative_to(ROOT)),link=link))
    assert not broken,broken
    v=dict(complete=True,attempts=262,profiles=52,comparisons=26,protected_files=len(before),
        changed_protected_files=changed,changed_execution_sources=sources_changed,
        frozen_predictions_before_exact=frozen_order,work_and_prediction_pairs=work_prediction_pairs,
        worker_overlap_detected=False,clean_guardians=guard_count,active_control_finished=active['finished'],
        mc_path_steps=W,remizov_operator_applications=S,peak_process_rss_mib=peak,
        attempts_longer_than_old_60s=len(longer60),longer60_attempt_ids=longer60,
        observed_accuracy_passes={f'{mode}/{method}':n for (mode,method),n in error_passes.items()},
        no_attempt_phase_total_duration_cutoffs=True,historical_results_unchanged=True,
        status_before_sha256=sha(OUT/'PROJECT_STATUS.before.md'),status_after_sha256=sha(ROOT/'PROJECT_STATUS.md'),
        all_trial_errors_independently_recomputed=True,all_fixed_work_counts_verified=True,
        source_sha256=plan['source_sha256'],broken_links=broken,new_production1000_solves=0,new_ml_runs=0)
    (OUT/'verification.json').write_text(json.dumps(v,indent=2,sort_keys=True,ensure_ascii=False)+'\n')
    print(json.dumps({k:x for k,x in v.items() if k not in ('source_sha256','longer60_attempt_ids')},indent=2,ensure_ascii=False))


if __name__=='__main__':main()
