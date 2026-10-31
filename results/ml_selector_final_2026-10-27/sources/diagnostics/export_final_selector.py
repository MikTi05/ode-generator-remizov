"""Export the independently reproducible train fit as a self-contained release."""
from __future__ import annotations
import copy,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from ml.revision_evidence import read,sha,ORIGINAL,CONTRACT
from ml.revision_selector import ResearchRevision
from ml.remizov_features import CERT_FIELDS,VERSION
from ml.final_selector import FinalResearchSelector
from diagnostics.prepare_development8 import write_json,utc


def export(out):
    from diagnostics.report_train_revision import report,markdown
    from diagnostics.verify_train_revision import verify
    training=out/'training';metrics=report(training);verify(training);markdown(training,metrics)
    model=out/'model';model.mkdir(exist_ok=False)
    registry=read(training/'registry.json');registry=copy.deepcopy(registry)
    paths={}
    for name,source in registry['final_heads'].items():
        path=model/'heads'/name;shutil.copytree(training/source,path);paths[name]=str(path.relative_to(model))
    registry.update(final_heads=paths,release_schema='final_research_selector_v1',implementation_complete=True,quality_independently_confirmed=False,
        feature_version=VERSION,inference_code_sha256=sha(ROOT/'ml/final_selector.py'),engine_code_sha256=sha(ROOT/'ml/revision_selector.py'),
        profile_context=read(CONTRACT/'plan.json')['profiles'],input_grid_sha256=sha_grid(),
        timing_context=dict(scope='warm_end_to_end_method_wall_v1',platform='macOS-26.6.2-arm64-arm-64bit',workers=1,threads=1,
            numpy='2.2.6',scipy='1.15.3',common_load_and_JIT='separate from method wall'),
        interpretation='probability targets E_grid33<=.01; train-selected near-best-probability admissibility, not an actual accuracy certificate',
        training_registry_sha256=sha(training/'registry.json'))
    write_json(model/'registry.json',registry,exclusive=True)
    write_json(model/'artifact_sha256.json',{str(p.relative_to(model)):sha(p) for p in sorted(model.rglob('*')) if p.is_file()},exclusive=True)
    served=FinalResearchSelector(model);reference=ResearchRevision(training)
    with np.load(training/'features_train.npz',allow_pickle=False) as z:X=z['X']
    assert served.model.predict_features(X)==reference.predict_features(X)
    previous=ROOT/'results/ml_train_revision_2026-10-27'
    assert read(training/'outer_oof.json')==read(previous/'outer_oof.json')
    assert read(training/'registry.json')['final_policy']==read(previous/'registry.json')['final_policy']
    # Input-only example comes from TRAIN, before any new final-test inputs exist.
    row=read(training/'accuracy.json')[0];pid=row['equation_id'];meta={}
    for line in (training/'training_inputs/all_methods_equations_v1_metadata.jsonl').open():
        m=json.loads(line)
        if m['equation_id']==pid:meta=m;break
    allowed={k:meta[k] for k in CERT_FIELDS};allowed['serialized_ast']={k:meta['serialized_ast'][k] for k in ('a_raw','b_raw','c_raw')}
    with np.load(ORIGINAL/'all_methods_equations_v1_ml.npz',allow_pickle=False) as z:
        i=next(i for i,v in enumerate(z['equation_id']) if str(v)==pid);assert str(z['split'][i])=='train'
        payload=dict(coefficients=z['X'][i].tolist(),x_grid=z['x_grid'].tolist(),coefficient_metadata=allowed,domain='all_methods_v1')
    write_json(out/'example_input.json',payload,exclusive=True);result=served.predict(payload['coefficients'],payload['x_grid'],allowed)
    write_json(out/'example_prediction.json',result,exclusive=True)
    write_json(out/'release_freeze.json',dict(frozen_utc=utc(),before_final_holdout_inputs=True,model_registry_sha256=sha(model/'registry.json'),
        model_manifest_sha256=sha(model/'artifact_sha256.json'),inference_code_sha256=sha(ROOT/'ml/final_selector.py'),engine_code_sha256=sha(ROOT/'ml/revision_selector.py'),
        feature_code_sha256=sha(ROOT/'ml/remizov_features.py'),policy=registry['final_policy'],training_reproduced_bitwise_OOF=True,final_heads_predictions_reproduced_exactly=True,
        production_blocked=True,implementation_complete=True,independent_quality_confirmation_pending=True),exclusive=True)
    print(dict(exported=True,profile=result['selected_profile'],train_OOF_reproduced=True))


def sha_grid():
    import hashlib
    with np.load(ORIGINAL/'all_methods_equations_v1_ml.npz',allow_pickle=False) as z:return hashlib.sha256(z['x_grid'].tobytes()).hexdigest()


if __name__=='__main__':export(Path(sys.argv[1]).resolve())
