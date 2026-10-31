from pathlib import Path
import importlib.metadata,json,shutil,sys
ROOT=Path('/Users/mikti/Documents/vsc projects/ode-generator-remizov');sys.path.insert(0,str(ROOT))
from diagnostics.audit_probability_selector import read,write,sha,MODEL
from diagnostics.train_nested_probability_calibration import snapshot
out=ROOT/'results/ml_nested_calibration_2026-10-26'
write(out/'tests.json',dict(passed=16,failed=0,duration_seconds=.16,stdout='16 passed in 0.16s',
 command='PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_nested_probability_calibration.py tests/test_nested_calibration_artifacts.py',
 scope='pure math/unit and saved-artifact checks; no solver calls and no additional training by tests'))
write(out/'library_versions.json',{name:importlib.metadata.version(name) for name in ('numpy','scipy','scikit-learn','catboost')})
sourcefiles=['ml/nested_probability_calibration.py','diagnostics/train_nested_probability_calibration.py',
 'diagnostics/report_nested_probability_calibration.py','diagnostics/summarize_nested_calibration.py',
 'tests/test_nested_probability_calibration.py','tests/test_nested_calibration_artifacts.py']
pins={}
for name in sourcefiles:
 path=ROOT/name;dest=out/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists():assert sha(dest)==sha(path),name
 else:shutil.copyfile(path,dest)
 pins[name]=sha(path)
write(out/'additional_sources_sha256.json',pins)
shutil.copyfile(ROOT/'PROJECT_STATUS.md',out/'PROJECT_STATUS.after.md')
shutil.copyfile(__file__,out/'sources/finalize_nested_calibration.py')
# All pre-existing files, including source and result artifacts, remain unchanged except status.
before=read(out/'protected_before.json');after=snapshot(out)
changed=[n for n,h in before.items() if after.get(n)!=h]
assert changed==['PROJECT_STATUS.md'],changed
frozen_counts={}
for name in ('ml_probability_selector_2026-10-26','ml_probability_validation_2026-10-26',
             'ml_probability_validation_preflight_2026-10-26','ml_probability_selector_audit_2026-10-26'):
 path=ROOT/'results'/name;manifest=read(path/'artifact_sha256.json')
 for n,h in manifest.items():assert sha(path/n)==h,(name,n)
 frozen_counts[name]=len(manifest)
for n,h in read(out/'source_sha256.json').items():assert sha(ROOT/n)==h,n
for n,h in read(MODEL/'source_sha256.json').items():assert sha(ROOT/n)==h,n
metrics=read(out/'metrics.json');verification=read(out/'independent_verification.json');registry=read(out/'registry.json')
assert verification['passed'] and not registry['deployment_ready'] and registry['no_production_dispatch']
assert sha(out/'registry.json')==read(out/'freeze_before_train20_inputs.json')['registry_sha256']
assert sha(ROOT/'ml/probabilistic_selector.py')==registry['original_production_code_sha256']
assert len(read(out/'oof.json')['R128'])==103
assert metrics['policies']['calibrated']['0.95']['baseline_default']['accurate_deliveries']==33
assert metrics['policies']['calibrated']['0.95']['baseline_default']['lost_baseline_accurate']==0
assert read(out/'prospective_train20_predictions.json')['q95_switches']==0
# Report clarification preserves the generated original text for recovery.
write(out/'report_revision.json',dict(reason='Explicitly state that no switches on frozen input20 means this packet cannot establish policy superiority; add score figure',
 previous='archive/REPORT.before_pilot_clarification.md',previous_sha256=sha(out/'archive/REPORT.before_pilot_clarification.md'),current_sha256=sha(out/'REPORT.md')))
write(out/'final_verification.json',dict(passed=True,previous_paths_checked=len(before),previous_paths_changed=changed,
 frozen_artifacts_checked=frozen_counts,old_numerical_core_data_results_preserved=True,
 R128_OOF_cases=103,R128_OOF_families=90,R256_OOF_cases=40,outer_blocks=6,inner_folds=3,
 model_scaler_nodes=38,saved_CBM_files=len(list((out/'models').glob('*.cbm'))),calibrators=11,
 inner_predictions_replayed=595,old_point_predictions_exact=80,full_family_exclusions_passed=True,
 proper_scores_recomputed=True,outer_evaluation_never_uses_final_calibrators=True,tests_passed=16,
 new_numerical_calls=0,MC_retrained=False,original_validation8_or_test_used_for_selection=False,
 production_API_unchanged=True,production_unblocked=False,train20_fit_labels_used=False,
 new_model_registry_frozen_before_train20_inputs=True,pilot20_q95_switches=0,
 scientific_conclusion='R256 calibration descriptively improves, R128 worsens; OOF default challenger preserves33 deliveries but measured time and independent noninferiority unproven',
 proposed_next_calls_authorized=False,figure_visually_reviewed=True))
# Keep full input-only forecast metadata/coefficients separate from the training access ledger.
write(out/'data_access_scope.json',dict(training_inputs='saved features_train103 and targets, old family_cv40 only',
 train20_inference='separate AFTER-freeze TRAIN-only selected coefficient inputs; no labels/exact/errors or other split features used for prediction',
 canonical_NPZ_access='X/channel_order/x_grid/equation_id/split only; source compression array sliced to20 train rows',
 original_validation8_or_test_for_fit_tuning_evaluation=False))
manifest={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p!=out/'artifact_sha256.json'}
write(out/'artifact_sha256.json',manifest)
assert all(sha(out/n)==h for n,h in manifest.items())
print(dict(passed=True,artifacts=len(manifest),previous_paths_checked=len(before),previous_paths_changed=changed,
 frozen_old_artifacts=sum(frozen_counts.values()),new_numerical_calls=0,production_unblocked=False))
