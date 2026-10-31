from pathlib import Path
import json,shutil,sys
ROOT=Path('/Users/mikti/Documents/vsc projects/ode-generator-remizov');sys.path.insert(0,str(ROOT))
from diagnostics.evaluate_selector_generalization import snapshot,input_metadata
from diagnostics.audit_probability_selector import read,write,sha
out=ROOT/'results/ml_selector_generalization_plan_2026-10-26'
metadata=input_metadata();signatures=sorted({m['structure_signature'] for m in metadata.values()})
write(out/'sampling_rules.json',dict(execution_authorized=False,generate_now=False,pool_seed=2488000901,pool_families=40,
 samples_per_family=1,config='copy unchanged production all_methods_v1 parameter ranges, validation/ML grid generation settings; only new seed/family/sample count and new output directory',
 only_train_role_eligible=True,minimum_known_excluded_topology_signatures=signatures,
 full_history_exposure_audit_required=True,exclusion_scope='all original TRAIN and8 revealed validation signatures; no test labels or features used',
 target_families=8,complexity_order=['L1','L2','L3','L4'],quota_per_complexity=2,
 deterministic_order='two rounds of L1,L2,L3,L4; among still eligible candidates at that complexity choose least represented regime, then lexicographic SHA256(fresh_MC8_v1|dataset_sha|structure_signature|canonical_ID); regime ties by that same hash',
 input_only_selection=True,numerical_accuracy_time_outcomes_used_for_selection=False,
 stop_if_insufficient_unique_topologies=True,no_automatic_pool_expansion_or_replacing_failures=True,
 family_identity='dataset_sha256 + structure_signature; canonical family/equation strings alone may repeat across cohorts',
 freeze_before_numerics=['exact8 IDs/AST/input hashes/profiles/analytic G,T,N','all frozen predictions and primary q=.95 decisions',
 'MC seed entropy248800080100..103 nonreuse','per-case memory/work/watchdogs and total cap approval'],
 numerical_profiles='same R128/R256/MC1 contract, grid33[-1,1],eps.01; no parameter or implementation change',
 MC_resource_guard='max_path_steps must cover actual33*4096*N; resource guards are frozen case-by-case in preflight, never changed reactively after a failure',
 final_policy_test_gate='a new frozen version must have meaningful action coverage; current MC action globally unreachable, so current freeze cannot substantiate three-method selection',
 utility_pilot_scope='all-profile feasibility data even if policy selects no MC; not a test of MC-selection reliability, no fitting on packet in this task'))
write(out/'tests.json',dict(passed=7,failed=0,duration_seconds=.23,stdout='7 passed in 0.23s',
 command='PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_selector_generalization.py',
 scope='leaf-bound math, cost/accuracy/abstention, inert budget and AST no-fit/no-ODE checks'))
write(out/'cost_scope_notes.json',dict(primary_inference='actual current calibrated feature+heads+calibration+primary q95 choice timer',
 method_cost='saved median4 per selected method, same validation cohort, includes inaccurate answers',
 secondary_variants='shared primary diagnostic-stack inference overhead; not independent latency benchmarks of raw/legacy API',
 production_guard='API remains blocked; no successful delivery and cost-per-success undefined, not a performance win',
 retrospective_only=True))
extra={}
for name in ('diagnostics/evaluate_selector_generalization.py','diagnostics/report_selector_generalization.py','tests/test_selector_generalization.py'):
 path=ROOT/name;dest=out/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists():assert sha(dest)==sha(path),name
 else:shutil.copyfile(path,dest)
 extra[name]=sha(path)
write(out/'additional_sources_sha256.json',extra)
shutil.copyfile(__file__,out/'sources/finalize_generalization.py')
shutil.copyfile(ROOT/'PROJECT_STATUS.md',out/'PROJECT_STATUS.after.md')
for name,h in read(out/'source_sha256.json').items():assert sha(ROOT/name)==h,name
frozen={}
for name in ('ml_probability_selector_2026-10-26','ml_nested_calibration_2026-10-26','ml_probability_validation_2026-10-26','ml_switch5_r_timing_2026-10-26'):
 folder=ROOT/'results'/name;manifest=read(folder/'artifact_sha256.json')
 for n,h in manifest.items():assert sha(folder/n)==h,(name,n)
 frozen[name]=len(manifest)
before=read(out/'protected_before.json');after=snapshot(out);changed=[n for n,h in before.items() if after.get(n)!=h]
assert changed==['PROJECT_STATUS.md'],changed
assert read(out/'independent_verification.json')['passed']
assert not read(out/'prospective_experiment.json')['execution_authorized']
assert read(out/'model_reachability.json')['MC_q95_action_impossible']
assert read(out/'retrospective_results.json')['policies']['0.95']['baseline_default']['actions']=={'R256':8}
write(out/'final_verification.json',dict(passed=True,previous_files_checked=len(before),previous_files_changed=changed,
 frozen_model_result_artifacts_checked=frozen,new_solver_calls=0,new_MC_calls=0,new_ML_fits=0,new_calibrator_fits=0,
 new_generated_datasets=0,test_used=False,old8_validation_explicitly_retrospective=True,primary_policy_frozen_and_unchanged=True,
 primary_actions_R256=8,primary_correct=7,primary_violations=1,primary_abstentions=0,method_saving_percent=0.,
 MC_model_global_upper_below_q95=True,MC_true_accuracy_or_uselessness_not_claimed=True,never_fit_input_scan=250,
 prospective_pilot_calls_proposed=96,prospective_budget_cap_unapproved=5460,final_test_not_ready=True,
 production_still_blocked=True,tests_passed=7,status_backed_up=True))
manifest={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p!=out/'artifact_sha256.json'}
write(out/'artifact_sha256.json',manifest);assert all(sha(out/n)==h for n,h in manifest.items())
print(dict(passed=True,artifacts=len(manifest),old_files_checked=len(before),old_files_changed=changed,
 frozen_artifacts_checked=sum(frozen.values()),new_solver_calls=0))
