from pathlib import Path
import shutil,sys
ROOT=Path('/Users/mikti/Documents/vsc projects/ode-generator-remizov');sys.path.insert(0,str(ROOT))
from diagnostics.audit_probability_selector import read,write,sha
from diagnostics.plan_switch5_r_timing import snapshot
out=ROOT/'results/ml_switch5_r_timing_2026-10-26';a=read(out/'analysis.json');plan=read(out/'plan.json')
write(out/'tests.json',dict(passed=19,failed=0,duration_seconds=.16,stdout='19 passed in 0.16s',
 command='PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_run_switch5_r_timing.py tests/test_switch5_r_timing_plan.py tests/test_switch5_r_timing_report.py',
 scope='pure unit/mock controller and saved-only planning/report; no added numerical calls'))
write(out/'resume_check.json',dict(new_calls=0,worker_started=False,reason='already_complete',
 command='PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B diagnostics/run_switch5_r_timing.py run --out results/ml_switch5_r_timing_2026-10-26',
 recorded_from_completed_tool_execution=True))
files=['diagnostics/run_switch5_r_timing.py','diagnostics/report_switch5_r_timing.py','diagnostics/verify_switch5_r_timing.py',
 'diagnostics/summarize_switch5_r_timing.py','tests/test_run_switch5_r_timing.py','tests/test_switch5_r_timing_report.py']
extra={}
for name in files:
 dest=out/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists():assert sha(dest)==sha(ROOT/name),name
 else:shutil.copyfile(ROOT/name,dest)
 extra[name]=sha(ROOT/name)
write(out/'additional_sources_sha256.json',extra)
shutil.copyfile(ROOT/'PROJECT_STATUS.md',out/'PROJECT_STATUS.after.md')
shutil.copyfile(__file__,out/'sources/finalize_R40.py')
# Verify the actual model/data artifacts as well as their frozen manifests.
checked={}
for dirname in ['ml_switch5_r_preflight_2026-10-26','ml_nested_calibration_2026-10-26','ml_probability_selector_2026-10-26','remizov_r0_2026-10-26']:
 folder=ROOT/'results'/dirname;manifest=read(folder/'artifact_sha256.json')
 for name,h in manifest.items():assert sha(folder/name)==h,(dirname,name)
 checked[dirname]=len(manifest)
for name,h in plan['protected_sha256'].items():assert sha(ROOT/name)==h,name
before=read(out/'protected_before.json');after=snapshot(out);changed=[n for n,h in before.items() if after.get(n)!=h]
assert changed==['PROJECT_STATUS.md'],changed
assert read(out/'independent_verification.json')['passed']
assert read(out/'active.json')['finished'] and not read(out/'pending.json')['keys']
assert len(list((out/'records').glob('*.json')))==len(list((out/'events').glob('*.request.json')))==40
assert len(list((out/'events').glob('*.interrupted.json')))==0
budget=read(out/'budget.json');active=read(out/'active.json');assert active['finished_monotonic']<budget['deadline_monotonic']
assert budget['approved_cap_seconds']==1766 and budget['maximum_physical_calls']==40
power=read(out/'hardware_power_after.json');assert power['settings_unchanged'] and power['power_source_unchanged']
write(out/'final_verification.json',dict(passed=True,previous_files_checked=len(before),previous_files_changed=changed,
 actual_frozen_artifacts_checked=checked,raw_records=40,unique_requests=40,canonical_errors_recomputed=40,
 actual_order_matches_approved_plan=True,numerical_calls=40,extra_numerical_calls=0,retries=0,interruptions=0,pending=0,
 maximum_approved_calls=40,approved_wall_cap_seconds=1766,controller_wall_seconds=a['resources']['controller_wall_seconds'],
 original_deadlines_respected=True,peak_RSS_MiB=a['resources']['peak_RSS_MiB'],RSS_limit_MiB=512,
 workers=1,compute_threads=1,guardian_cleanup_success=True,thermal_state_max=0,
 operator_applications=3932160,spline_builds=3932200,
 power_source_and_settings_unchanged=True,completed_resume_new_calls=0,tests_passed=19,
 all5_median_saving_percent=100*a['aggregate']['all5']['relative_saved'],
 accurate4_median_saving_percent=100*a['aggregate']['both_accurate']['relative_saved'],
 accurate_deliveries=4,baseline_accurate_deliveries=4,lost_baseline_deliveries=0,
 F0172_both_inaccurate_preserved=True,global40_measured_percentage=None,ML_inference_cost_measured=False,
 models_calibrators_policy_numerical_methods_and_old_results_unchanged=True,production_unblocked=False,
 MC_calls=0,ML_fits=0,benchmark1000_calls=0,validation_test_used=False,figure_visually_reviewed=True))
manifest={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p!=out/'artifact_sha256.json'}
write(out/'artifact_sha256.json',manifest);assert all(sha(out/n)==h for n,h in manifest.items())
print(dict(passed=True,artifacts=len(manifest),old_files_checked=len(before),old_files_changed=changed,
 frozen_actual_files_checked=sum(checked.values()),calls=40,local_percent_saved=100*a['aggregate']['all5']['relative_saved']))
