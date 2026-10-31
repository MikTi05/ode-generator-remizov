from pathlib import Path
import csv,json,shutil
from collections import Counter
import sys
ROOT=Path('/Users/mikti/Documents/vsc projects/ode-generator-remizov');sys.path.insert(0,str(ROOT))
from diagnostics.audit_probability_selector import read,write,sha,MODEL,VALIDATION,PREFLIGHT
out=ROOT/'results/ml_probability_selector_audit_2026-10-26';archive=out/'archive';archive.mkdir(exist_ok=False)
report=out/'REPORT.md';shutil.copyfile(report,archive/'REPORT.before_clarification.md')
text=report.read_text()
old='не утверждение о статистически достаточной калибровке. По одному R128 и R256 для детерминированной '
new='по одному семейству в каждой из20 ячеек regime×complexity; не утверждение о статистически достаточной калибровке. По одному R128 и R256 для детерминированной '
assert old in text;text=text.replace(old,new,1)
old='до этого audit. Первые метрики — точные доставки, нарушения среди выбранных, coverage, '
new='до этого audit. Never-fit относится только к текущим шести головам/scalers: это не доказательство отсутствия исторического раскрытия результатов. Перед признанием audit независимым проверить всю историю сопоставимых профильных outcomes и использования семейств; при раскрытии нужен новый заранее разделённый набор. Первые метрики — точные доставки, нарушения среди выбранных, coverage, '
assert old in text;text=text.replace(old,new,1);report.write_text(text)
write(out/'report_clarification.json',dict(reason='Clarify20 input strata and distinguish never-fit from historically unseen',old_sha256=sha(archive/'REPORT.before_clarification.md'),new_sha256=sha(report)))
revisions=[]
source_hashes={}
for name in ('diagnostics/audit_probability_selector.py','diagnostics/report_probability_selector_audit.py','tests/test_probability_selector_audit.py'):
 dest=out/'sources'/name
 if dest.exists() and sha(dest)!=sha(ROOT/name):
  previous=archive/('initial_'+Path(name).name);shutil.copyfile(dest,previous)
  revisions.append(dict(source=name,initial_sha256=sha(dest),final_sha256=sha(ROOT/name),backup=str(previous.relative_to(out)),reason='No numerical/model changes: simplify baseline call, add statistical tests, clarify report scope'))
 dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest);source_hashes[name]=sha(ROOT/name)
write(out/'source_revisions.json',revisions);write(out/'additional_sources_sha256.json',source_hashes)
write(out/'tests.json',dict(passed=15,failed=0,duration_seconds=.52,stdout='15 passed in 0.52s',
 command='PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_probability_selector_audit.py tests/test_probability_validation_report.py',
 scope='pure unit and saved-only accounting/statistics/AST checks; no solver or ML fit, no dataset-test'))
plan=read(out/'acquisition_plan.json');registry=read(MODEL/'registry.json')
fit=set().union(*(set(h['metadata']['families']) for h in registry['heads'].values()))
metadata=ROOT/read(VALIDATION/'plan.json')['data_dir']/'all_methods_equations_v1_metadata.jsonl'
train=[]
with metadata.open() as f:
 for line in f:
  row=json.loads(line)
  if row['split']=='train':train.append({k:row[k] for k in ('equation_id','family_id','regime','complexity_level','expression_hash')})
byfamily={r['family_id']:r for r in train};available=set(byfamily)-fit;chosen={r['family_id'] for r in plan['cases']};reserved=sorted(available-chosen)
assert len(available)==50 and len(chosen)==20 and len(reserved)==30
cells=lambda rows:Counter(r['regime']+'|'+r['complexity_level'] for r in rows)
assert len(cells(plan['cases']))==20 and set(cells(plan['cases']).values())=={1}
frame=dict(original_train_families=140,never_fit_current_heads=50,proposed_new_train_families=20,reserved_families=reserved,
 proposed_stratum_counts=dict(cells(plan['cases'])),never_fit_stratum_counts=dict(cells(byfamily[g] for g in available)),
 historical_exposure_audit_required=True,never_fit_is_not_historically_unseen=True,
 reserve_scope='Future internal family audit after a new freeze; no model/scaler/calibrator/threshold use before audit; check full historical outcome exposure first, else generate a new separately split cohort',
 no_test_outcome_use=True)
write(out/'sampling_frame.json',frame)
write(out/'recommendations.json',dict(current_changes_to_models_policy=None,
 next_offline_task='Future authorized ML task: full R128 family-OOF on103 saved cases/90 families; prespecified simple regularized sigmoid calibration in nested family splits; no tuning on eight validation families',
 calibration_limits='Monotone calibration cannot repair F0041/F0189 ranking; MC high-p regime unsupported by3 ever-success train families',
 acquisition='20 new never-fit original-train families, one per20 regime/complexity cells, fixed IDs; max120 calls as development only; scenario4726.9s, separate proposed cap9754s, not approved',
 independent_check='Reserve remaining30 never-fit families before new training/calibration; full historical exposure audit required; internal diagnostic only, not final test or95pct certification',
 q='Unchanged .5/.8/.95; caller risk q=.95 remains an unverified requirement, not a tuned optimum',
 metrics=['accurate deliveries/all requests','inaccurate/selected','abstention rate','total warm cost/accurate deliveries','MC choices and their fresh-seed outcomes'],
 formal_evidence='59/59 independent accepted post-freeze family outcomes for one-sided95pct lower bound>=.95, or92/93 with one failure; not a universal calibration-training-size rule',
 stop='No automatic expansion after20 families to manufacture MC support; expose absent success/winner regimes and budget separately',
 new_numerical_calls=0,new_ML_fits=0,new_calibrator_fits=0,test_used=False))
shutil.copyfile(ROOT/'PROJECT_STATUS.md',out/'PROJECT_STATUS.after.md')
shutil.copyfile(__file__,out/'sources'/'finalize_audit.py')
checked={}
for directory in (MODEL,VALIDATION,PREFLIGHT):
 manifest=read(directory/'artifact_sha256.json')
 for name,h in manifest.items():assert sha(directory/name)==h,(directory.name,name)
 checked[directory.name]=len(manifest)
for name,h in read(out/'source_sha256.json').items():assert sha(ROOT/name)==h,name
protected=read(VALIDATION/'plan.json')['protected_sha256']
for name,h in protected.items():assert sha(ROOT/name)==h,name
assert sha(ROOT/'PROJECT_STATUS.md')==sha(out/'PROJECT_STATUS.after.md')
assert sha(ROOT/'PROJECT_STATUS.md')!=sha(out/'PROJECT_STATUS.before.md')
write(out/'final_verification.json',dict(passed=True,frozen_manifests_checked=checked,protected_source_inputs_checked=len(protected),
 OOF_families=40,validation_families=8,independent_scaler_means=18,independent_saved_error_checks=96,
 tests_passed=15,new_calls=0,ML_fits=0,calibrator_fits=0,test_used=False,models_policy_q_unchanged=True,
 existing_results_unchanged=True,proposed_train_cases=20,proposed_input_strata=20,reserved_train_families=30,
 prior_record_scan='prior_grid_record_inventory.json; no planned/reserve33-grid records found in scanned records directories; full history audit still required',
 reports_figures_visually_reviewed=True,status_backup_and_after_saved=True))
manifest={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p!=out/'artifact_sha256.json'}
write(out/'artifact_sha256.json',manifest)
assert all(sha(out/name)==h for name,h in manifest.items())
print(dict(passed=True,artifacts=len(manifest),frozen_artifacts_checked=sum(checked.values()),protected=len(protected),reserved=30,status_updated=True))
