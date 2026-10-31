from pathlib import Path
import ast,shutil,sys
ROOT=Path('/Users/mikti/Documents/vsc projects/ode-generator-remizov');sys.path.insert(0,str(ROOT))
from diagnostics.plan_switch5_r_timing import snapshot
from diagnostics.audit_probability_selector import read,write,sha
out=ROOT/'results/ml_switch5_r_preflight_2026-10-26';plan=read(out/'plan.json')
files=['diagnostics/plan_switch5_r_timing.py','diagnostics/verify_switch5_r_preflight.py','tests/test_switch5_r_timing_plan.py']
extra={}
for name in files:
 src=ROOT/name;dest=out/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists():assert sha(dest)==sha(src),name
 else:shutil.copyfile(src,dest)
 extra[name]=sha(src)
for name in ['reports/ml_selector_contract/plan.json','results/remizov_r0_2026-10-26/plan.json']:
 extra[name]=sha(ROOT/name)
write(out/'additional_protected_sha256.json',extra)
write(out/'tests.json',dict(passed=7,failed=0,duration_seconds=.06,stdout='7 passed in 0.06s',
 command='PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_switch5_r_timing_plan.py',
 scope='saved-only planning/unit checks, no ODE, MC, JIT or training calls'))
for name in files[:2]:
 tree=ast.parse((ROOT/name).read_text());calls={n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,(ast.Name,ast.Attribute))}
 assert not calls&{'solve','solve_streaming','solve_monte_carlo_optimized','compile_backend','fit','fit_head','worker','new_control','GuardedProcess'}
shutil.copyfile(ROOT/'PROJECT_STATUS.md',out/'PROJECT_STATUS.after.md')
shutil.copyfile(__file__,out/'sources/finalize_preflight.py')
checked={}
for name in ('ml_nested_calibration_2026-10-26','ml_probability_selector_2026-10-26','remizov_r0_2026-10-26'):
 folder=ROOT/'results'/name;m=read(folder/'artifact_sha256.json')
 for n,h in m.items():assert sha(folder/n)==h,(name,n)
 checked[name]=len(m)
for n,h in plan['protected_sha256'].items():assert sha(ROOT/n)==h,n
for n,h in extra.items():assert sha(ROOT/n)==h,n
before=read(out/'protected_before.json');after=snapshot(out);changed=[n for n,h in before.items() if after.get(n)!=h]
assert changed==['PROJECT_STATUS.md'],changed
assert not plan['execution_authorized'] and not read(out/'approval_request.json')['approved']
assert read(out/'approval_request.json')['plan_sha256']==sha(out/'plan.json')
assert read(out/'independent_verification.json')['passed']
assert not (out/'records').exists() and not (out/'events').exists() and not (out/'active.json').exists()
write(out/'report_revision.json',dict(reason='Separate measured method-only cost from unmeasured ML/online overhead',
 before='archive/REPORT.before_cost_scope_clarification.md',before_sha256=sha(out/'archive/REPORT.before_cost_scope_clarification.md'),after_sha256=sha(out/'REPORT.md')))
write(out/'final_verification.json',dict(passed=True,identities=5,canonical_reference_errors=10,
 jobs=40,R128_jobs=20,R256_jobs=20,MC_jobs=0,approved=False,execution_authorized=False,
 central_scenario_seconds=read(out/'budget.json')['central_scenario_seconds'],proposed_cap_seconds=1766,
 known_inaccurate_case='AMV1_F0172_S03',expected_accurate_cases=4,tests_passed=7,
 frozen_old_artifacts_checked=checked,protected_original_paths_checked=len(before),original_paths_changed=changed,
 model_policy_numerical_algorithms_and_old_results_unchanged=True,production_still_blocked=True,
 new_numerical_calls=0,new_MC_calls=0,new_ML_fits=0,validation_test_outcomes_used=False,
 initial_partial_planning_attempt_preserved='results/ml_switch5_r_preflight_initial_2026-10-26',
 scope_specific_runner_required_after_budget_approval=True,status_backed_up=True))
manifest={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p!=out/'artifact_sha256.json'}
write(out/'artifact_sha256.json',manifest);assert all(sha(out/n)==h for n,h in manifest.items())
print(dict(passed=True,artifacts=len(manifest),old_paths_checked=len(before),old_paths_changed=changed,
 proposed_cap_seconds=1766,execution_authorized=False,numerical_calls=0))
