"""Reproduce post-hoc saved-data diagnostics; no fitting or numerical calls."""
import json,statistics
from pathlib import Path
OUT=Path(__file__).resolve().parent.parent

def build():
 a=json.loads((OUT/'analysis.json').read_text());v=json.loads((OUT/'final_independent_verification.json').read_text());cases=a['cases']
 feasible=[c for c in cases if any(c['profiles'][p]['correct']==1 for p in ('R128','R256'))]
 miss=[];optimal=0.;chosen_cost=0.
 for c in feasible:
  ps=c['profiles'];best=min((p for p in ('R128','R256') if ps[p]['correct']==1),key=lambda p:ps[p]['seconds']);chosen=c['policies']['selector']['profile']
  assert ps[chosen]['correct']==1
  optimal+=ps[best]['seconds'];chosen_cost+=ps[chosen]['seconds']
  if best!=chosen:
   assert best=='R128' and chosen=='R256';miss.append(c['case']['equation_id'])
 mcids={r['case']['equation_id'] for r in a['MC_secondary']['measured_rows']};mc=[c for c in cases if c['case']['equation_id'] in mcids]
 times=[c['profiles']['MC1']['seconds'] for c in mc];errors=[c['profiles']['MC1']['error'] for c in mc]
 probs=[r['prediction']['probability_accuracy'] for r in a['MC_secondary']['measured_rows']]
 spent=sum(c['profiles'][p]['seconds'] for c in cases for p in ('R128','R256','MC1') if c['profiles'][p]['seconds'] is not None)
 s=dict(R_only_oracle=dict(scope='post-hoc measured R128/R256 only; MC40 unknown',feasible_families=len(feasible),selected_accurate=len(feasible),fastest_accurate_selected=len(feasible)-len(miss),missed_faster_R128=miss,known_feasible_best_method_cost=optimal,known_feasible_selected_method_cost=chosen_cost,known_feasible_cost_regret=chosen_cost-optimal,unresolvable_with_measured_R=[c['case']['equation_id'] for c in cases if c not in feasible]),
 MC20=dict(successes=sum(c['profiles']['MC1']['correct'] for c in mc),finite_violations=sum(c['profiles']['MC1']['violation'] for c in mc),errors_min_median_max=[min(errors),statistics.median(errors),max(errors)],seconds_min_median_max=[min(times),statistics.median(times),max(times)],mean_predicted_accuracy_probability=statistics.mean(probs),predicted_accuracy_range=[min(probs),max(probs)],warm_cost_R128_same20=sum(c['profiles']['R128']['seconds'] for c in mc),warm_cost_R256_same20=sum(c['profiles']['R256']['seconds'] for c in mc),observed_MC_faster_than_R128=sum(c['profiles']['MC1']['seconds']<c['profiles']['R128']['seconds'] for c in mc),observed_MC_faster_than_R256=sum(c['profiles']['MC1']['seconds']<c['profiles']['R256']['seconds'] for c in mc),observed_MC_admissible_winners=sum(c['single_realization_10percent_tie_profile']=='MC1' for c in mc),claims_about_fresh_seed_reliability=False),
 execution=dict(total_method_wall_seconds=spent,full_elapsed_seconds=v['elapsed_to_guardian_cleanup_seconds'],series_overhead_seconds=v['elapsed_to_guardian_cleanup_seconds']-spent,ML_feature_inference_proxy_seconds=a['primary']['selector']['total_warm_cost']-a['primary']['selector']['known_observed_method_spend']))
 target=OUT/'final_diagnostics.json'
 if target.exists():assert json.loads(target.read_text())==s,'saved diagnostics disagree'
 else:target.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
 return dict(passed=True,diagnostics_reproduced=True,R_feasible=len(feasible),missed_faster_R128=len(miss),MC_successes=s['MC20']['successes'],new_solver_calls=0)

if __name__=='__main__':print(json.dumps(build()))
