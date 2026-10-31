"""Audit and synthesize SAVED research only. Never fit or solve equations."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/ode-ml-closure-mpl')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
R = ROOT/'results/remizov_ml_v1_2026-10-25'
C = ROOT/'results/mc1000_x0_2026-10-25'
J = ROOT/'results/joint_grid100_2026-10-25'
F = ROOT/'results/final_grid5_2026-10-25'
DATA = ROOT/'data/all_methods_v1_1000_2026-10-18'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                    ensure_ascii=False, allow_nan=False)+'\n')


def snapshot(out):
    paths = []
    for base in ('data', 'src', 'ml', 'diagnostics', 'tests', 'results'):
        paths.extend(p for p in (ROOT/base).rglob('*') if p.is_file() and out not in p.parents
                     and '__pycache__' not in p.parts and p.name != '.DS_Store')
    paths.extend(ROOT.glob('*.py'))
    paths.append(ROOT/'MIGRATION_MANIFEST.json')
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))}


def registry(inputs, out):
    from ml.remizov_features import VERSION
    rm = read(R/'models/metadata.json')
    cm = read(C/'ml/models/metadata.json')
    profiles, artifacts = {}, {}
    def artifact(name, path, kind, transform=None):
        artifacts[name] = dict(path=str(path.relative_to(ROOT)), sha256=sha(path), kind=kind)
        if transform:
            artifacts[name]['transform'] = transform
        return name
    assert rm['selected_classifier'] == 'classifier_catboost'
    assert rm['selected_regressor'] == 'regressor_dummy_context'
    assert set(cm['classification'].values()) == set(cm['regression'].values()) == {'catboost'}
    warning = ['Exploratory surrogate; no accuracy certificate or validated prediction interval',
               'Original family holdout; historical test outcomes previously inspected numerically',
               'No out-of-distribution validation or portability to changed numerical parameters']
    rp = dict(n=128, Q=513, dx=.0125, padding=12., interpolation='cubic',
              boundary='taper', T_policy='legacy, independent of G', tail_eps=1e-8)
    mp = dict(backend='numba', h=1/128, T_policy='analytic G auto_T', tail_eps=1e-4,
              chunk=1024, time_discretization_bound=None)
    profiles['remizov_taper_grid33'] = dict(status='supported_exploratory',
        scope=dict(method='Remizov', target='E_inf<=0.01', x=[-1+j/16 for j in range(33)]),
        parameters=rp, failure_threshold=rm['failure_threshold'], warnings=warning+[
            'Historical taper has known catastrophic boundary artifacts; not cubic/zero',
            rm['runtime_warning']], artifacts={
            'p_accuracy_met': artifact('R_grid_classifier', R/'models/classifier_catboost.cbm', 'classifier'),
            'historical_solver_seconds': artifact('R_grid_time', R/'models/regressor_dummy_context.json', 'context_median')})
    profiles['remizov_taper_x0'] = dict(status='supported_exploratory',
        scope=dict(method='Remizov', target='abs point error<=0.01; log10 error', x=[0.]),
        parameters=rp, warnings=warning+['Extracted node of historical grid33 solve; no point-only runtime model'],
        artifacts={
            'p_accuracy_met': artifact('R_point_classifier', C/'ml/models/remizov_point_accuracy_catboost.cbm', 'classifier'),
            'log10_absolute_error': artifact('R_point_error', C/'ml/models/remizov_point_log_error_catboost.cbm', 'regressor', 'pow10')})
    profiles['mc_x0_single1024'] = dict(status='supported_exploratory',
        scope=dict(method='Monte Carlo', target='probability of observed error<=0.01 for one seed', x=[0.]),
        parameters=dict(mp, M_per_point_per_seed=1024, seeds=1), warnings=warning+[
            'Probability is not a lower reliability bound; h bias unbounded'], artifacts={
            'p_accuracy_met': artifact('MC_single_classifier', C/'ml/models/mc_single_seed_accuracy_catboost.cbm', 'classifier')})
    profiles['mc_x0_mean4x1024'] = dict(status='supported_exploratory',
        scope=dict(method='Monte Carlo', target='log10 error of mean of four independent seed results', x=[0.]),
        parameters=dict(mp, M_per_point_per_seed=1024, seeds=4, total_paths_per_point=4096),
        warnings=warning+['Regression is for aggregate error; single-seed probability does not apply',
                          'No aggregate success classifier, MC runtime model, or h-bias bound'], artifacts={
            'log10_absolute_error': artifact('MC_mean_error', C/'ml/models/mc_aggregate_log_error_catboost.cbm', 'regressor', 'pow10')})
    for name, method, params in [
        ('remizov_zero_grid33_n128', 'Remizov', dict(rp, boundary='zero', T_policy='analytic G common T', tail_eps=1e-4)),
        ('remizov_zero_grid33_n256', 'Remizov', dict(rp, n=256, boundary='zero', T_policy='analytic G common T', tail_eps=1e-4)),
        ('mc_grid33_single1024', 'Monte Carlo', dict(mp, M_per_point_per_seed=1024, seeds=1)),
        ('mc_grid33_single4096', 'Monte Carlo', dict(mp, M_per_point_per_seed=4096, seeds=1)),
        ('mc_grid33_mean4x4096', 'Monte Carlo', dict(mp, M_per_point_per_seed=4096, seeds=4, total_paths_per_point=16384))]:
        profiles[name] = dict(status='unsupported_no_trained_model', parameters=params,
            scope=dict(method=method, x=[-1+j/16 for j in range(33)], target='E_inf<=0.01'),
            warnings=['No trained model for this profile; historical predictions are not transferred'])
    value = dict(schema='quality_registry_v1', feature_version=VERSION,
        feature_code_sha256=sha(ROOT/'ml/remizov_features.py'), feature_names=inputs['feature_names'],
        input_grid_sha256=inputs['grid_sha256'], selection='Frozen historical validation choices; no new selection',
        artifacts=artifacts, profiles=profiles)
    write(out/'registry.json', value)
    return value


def check_close(a, b):
    import numpy as np
    np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-12, equal_nan=True)


def audit(inputs, out):
    import numpy as np
    import pandas as pd
    import joblib
    from catboost import CatBoostClassifier, CatBoostRegressor
    from ml.remizov_features import coefficient_features, CERT_FIELDS
    from diagnostics.train_remizov_ml import classification_metrics, regression_metrics
    from diagnostics.train_mc1000_quality import metrics_class
    from sklearn.metrics import mean_absolute_error, r2_score
    from scipy.special import expit
    from ml.quality_prototype import QualityPrototype
    X = pd.DataFrame(inputs['X'], columns=inputs['feature_names'])
    rp = pd.read_csv(R/'predictions.csv'); cp = pd.read_csv(C/'ml/predictions.csv')
    rt = pd.read_csv(R/'targets.csv'); eq = pd.read_csv(C/'analysis/equations.csv')
    att = pd.read_csv(C/'analysis/attempts.csv'); ix = {pid:i for i,pid in enumerate(inputs['ids'])}
    for frame in (rp, cp, rt, eq):
        assert frame.equation_id.tolist() == list(inputs['ids'])
        assert np.array_equal(frame.split, inputs['splits'])
        assert np.array_equal(frame.family_id, inputs['families'])
    assert {s:int(sum(inputs['splits']==s)) for s in ('train','validation','test')} == {'train':700,'validation':150,'test':150}
    for field in ('family_id', 'structure_signature'):
        groups = {}
        for m in inputs['metadata'].values():
            groups.setdefault(m[field], set()).add(m['split'])
        assert all(len(v)==1 for v in groups.values())
    # Do not unpickle object feature_names from the old MC NPZ: schema has names.
    for folder in (R, C/'ml'):
        schema = read(folder/'feature_schema.json')
        assert schema['feature_names'] == list(X.columns)
        assert schema['input_grid_sha256'] == inputs['grid_sha256']
        with np.load(folder/'features.npz', allow_pickle=False) as z:
            assert np.array_equal(z['X'], inputs['X'])
            assert np.array_equal(z['equation_id'], inputs['ids'])
            assert np.array_equal(z['family_id'], inputs['families'])
            assert np.array_equal(z['split'], inputs['splits'])
    for actual, archived in [
        ('ml/remizov_features.py', R/'sources/remizov_features.py'),
        ('diagnostics/train_remizov_ml.py', R/'sources/train_remizov_ml.py'),
        ('ml/remizov_features.py', C/'sources/ml/remizov_features.py'),
        ('diagnostics/train_mc1000_quality.py', C/'sources/diagnostics/train_mc1000_quality.py')]:
        assert sha(ROOT/actual) == sha(archived)
    repro = read(C/'ml/reproducibility.json')
    for field, path in [('analysis_attempts_sha256', C/'analysis/attempts.csv'),
                        ('analysis_equations_sha256', C/'analysis/equations.csv'),
                        ('protocol_sha256', C/'analysis_protocol.json'),
                        ('feature_code_sha256', ROOT/'ml/remizov_features.py')]:
        assert sha(path) == repro[field]
    # Force every non-whitelisted metadata value to an unusable sentinel.
    with np.load(DATA/'all_methods_equations_v1_ml.npz', allow_pickle=False) as z:
        for i,pid in enumerate(inputs['ids']):
            m = inputs['metadata'][str(pid)]
            stripped = {key:m[key] for key in CERT_FIELDS}
            stripped['serialized_ast'] = {k:m['serialized_ast'][k] for k in ('a_raw','b_raw','c_raw')}
            for key in m:
                if key not in stripped:
                    stripped[key] = 'FORBIDDEN'
            feature = coefficient_features(z['X'][i], z['x_grid'], stripped)
            assert np.array_equal(list(feature.values()), inputs['X'][i])
    raw_counts = dict(remizov=0, mc=0)
    for row in rt.itertuples():
        p = ROOT/row.source_result
        assert sha(p)==row.source_result_sha256
        result = read(p)
        if row.target_known:
            error = float(np.max(np.abs(np.array(result['result']['f'])-result['y_exact'])))
            check_close(error, row.max_abs_error)
            assert int(error<=.01)==row.accuracy_met
            assert result['config']==read(R/'data_audit.json')['solver_config']
            assert len(result['target_x'])==33
            assert result['metadata']['expression_hash']==inputs['metadata'][row.equation_id]['expression_hash']
            k=result['target_x'].index(0.)
            point=abs(result['result']['f'][k]-result['y_exact'][k])
            check_close(point, eq.iloc[ix[row.equation_id]].remizov_error0)
            raw_counts['remizov'] += 1
        else:
            assert row.status=='worker_exception' and pd.isna(row.accuracy_met) and pd.isna(row.solver_seconds)
    for row in att.itertuples():
        p=ROOT/row.source; assert sha(p)==row.sha256
        result=read(p); j=ix[row.equation_id]
        assert row.family_id==inputs['families'][j] and row.split==inputs['splits'][j]
        assert result['expression_hash']==inputs['metadata'][row.equation_id]['expression_hash']
        assert result['target_x']==[0.] and result['status']=='completed'
        assert result['config']['M']==1024 and result['config']['dt']==1/128
        assert result['config']['tail_eps']==1e-4 and result['config']['T'] is None
        assert result['config']['seed']==row.seed and result['result']['completed_paths']==[1024]
        error=abs(result['result']['f'][0]-result['y_exact'])
        check_close(error, row.abs_error); assert int(error<=.01)==row.accuracy_met
        raw_counts['mc'] += 1
    assert raw_counts==dict(remizov=997,mc=4000)
    for pid, group in att.groupby('equation_id'):
        assert len(group)==4 and group.seed.nunique()==4
        check_close(abs(group.value.mean()-group.exact.iloc[0]),eq.iloc[ix[pid]].mc_aggregate_error)
    records=[]; models_checked=[]; scaler_checks=[]; linear_independent_checks=[]
    def reload(path, classifier):
        if path.suffix=='.cbm':
            model=CatBoostClassifier() if classifier else CatBoostRegressor()
            model.load_model(str(path))
        else:
            model=joblib.load(path)
        return model
    def replay(path, frame, expected, classifier, logarithmic, train_indices):
        model=reload(path,classifier)
        if hasattr(model,'feature_names_'):
            assert list(model.feature_names_)==list(frame.columns)
        kwargs={'thread_count':1} if path.suffix=='.cbm' else {}
        pred=model.predict_proba(frame,**kwargs)[:,1] if classifier else model.predict(frame,**kwargs)
        if logarithmic:
            pred=np.exp(pred)
        check_close(pred,expected)
        if hasattr(model,'steps'):
            scaler=model[0]; trainframe=frame.iloc[train_indices].to_numpy()
            check_close(scaler.mean_,np.mean(trainframe,axis=0))
            check_close(scaler.var_,np.var(trainframe,axis=0))
            assert int(scaler.n_samples_seen_)==len(train_indices)
            scaler_checks.append(dict(model=str(path.relative_to(ROOT)), rows=len(train_indices)))
            z=scaler.transform(frame); linear=model[1]
            coef=linear.coef_[0] if classifier else linear.coef_
            intercept=linear.intercept_[0] if classifier else linear.intercept_
            independent=np.einsum('ij,j->i',z,coef,optimize=False)+intercept
            if classifier:independent=expit(independent)
            elif logarithmic:independent=np.exp(independent)
            check_close(independent,pred)
            assert np.isfinite(independent).all()
            linear_independent_checks.append(dict(model=str(path.relative_to(ROOT)),
                max_abs_nonblas_difference=float(np.max(abs(independent-pred)))))
        models_checked.append(dict(path=str(path.relative_to(ROOT)),
            max_abs_prediction_difference=float(np.max(np.abs(pred-expected)))))
        return pred
    rm=read(R/'metrics.json'); meta=read(R/'models/metadata.json'); known=rt.target_known.to_numpy()
    train=np.flatnonzero((inputs['splits']=='train')&known)
    def metric_row(scope,task,model,split,metrics,selected):
        records.append(dict(scope=scope,task=task,model=model,split=split,selected=selected,**metrics))
    for key,spec in meta['models'].items():
        classifier=key.startswith('classifier_'); name=key.split('_',1)[1]
        frame=X.assign(planned_external_workers=rt.planned_external_workers) if spec['features']=='context' else X
        expected=rp[('p_success_' if classifier else 'seconds_')+name].to_numpy()
        if spec['kind']=='context_median':
            med=read(R/'models'/spec['path'])
            for w in (1,8):
                check_close(med[str(w)],np.median(rt.loc[(inputs['splits']=='train')&known&rt.planned_external_workers.eq(w),'solver_seconds']))
            pred=np.array([med[str(w)] for w in rt.planned_external_workers]);check_close(pred,expected)
            models_checked.append(dict(path=str((R/'models'/spec['path']).relative_to(ROOT)),max_abs_prediction_difference=float(np.max(abs(pred-expected)))))
        else:
            pred=replay(R/'models'/spec['path'],frame,expected,classifier,spec['transform']=='exp',train)
        for split in ('train','validation','test'):
            mask=(inputs['splits']==split)&known
            if classifier:
                for operating,stored in rm['classification'][name][split].items():
                    m=classification_metrics(rt.loc[mask,'accuracy_met'],pred[mask],stored['failure_threshold'])
                    for field in m:
                        check_close(m[field],stored[field])
                metric_row('historical_taper_grid33','accuracy',name,split,
                    {k:v for k,v in m.items() if not isinstance(v,np.ndarray)},key==meta['selected_classifier'])
            else:
                m=regression_metrics(rt.loc[mask,'solver_seconds'],pred[mask])
                for field in m:check_close(m[field],rm['regression'][name][split][field])
                metric_row('historical_taper_grid33','runtime',name,split,m,key==meta['selected_regressor'])
    cm=read(C/'ml/metrics.json'); seedix=np.array([ix[p] for p in att.equation_id])
    for category in ('classification','regression'):
        classifier=category=='classification'
        for task, models in cm[category].items():
            if classifier:
                indices=np.flatnonzero(known) if task.startswith('remizov') else seedix
                target=eq.loc[known,'remizov_accuracy0'].to_numpy() if task.startswith('remizov') else att.accuracy_met.to_numpy()
                tr=indices[inputs['splits'][indices]=='train']
            else:
                indices=np.flatnonzero(known) if task.startswith('remizov') else np.arange(1000)
                col='remizov_error0' if task.startswith('remizov') else 'mc_aggregate_error'
                target=np.log10(np.maximum(eq.iloc[indices][col].to_numpy(),1e-12))
                tr=indices[inputs['splits'][indices]=='train']
            selected=models['selected_by_validation_logloss' if classifier else 'selected_by_validation_mae']
            for kind, stored in models.items():
                if not isinstance(stored,dict):continue
                path=C/'ml/models'/f"{task}_{kind}{'.cbm' if kind=='catboost' else '.joblib'}"
                expected=cp[('p_' if classifier else 'logerror_')+task+'_'+kind].to_numpy()
                pred=replay(path,X,expected,classifier,False,tr)
                for split in ('train','validation','test'):
                    mask=inputs['splits'][indices]==split; ii=indices[mask]
                    if classifier:m=metrics_class(target[mask],pred[ii])
                    else:m=dict(n=len(ii),mae_log10=mean_absolute_error(target[mask],pred[ii]),
                        mae_error=mean_absolute_error(eq.iloc[ii][col],10.**pred[ii]),r2_log10=r2_score(target[mask],pred[ii]))
                    for field in m:
                        if m[field] is not None:check_close(m[field],stored[split][field])
                    metric_row('x0',task,kind,split,m,kind==selected)
    # Validation selection is rechecked, NEVER changed using test results.
    assert meta['selected_classifier']=='classifier_'+max(rm['classification'],key=lambda k:(rm['classification'][k]['validation']['validation_threshold']['roc_auc'],-rm['classification'][k]['validation']['validation_threshold']['logloss']))
    assert meta['selected_regressor']=='regressor_'+min(rm['regression'],key=lambda k:rm['regression'][k]['validation']['mae_seconds'])
    for category,selection,score in [('classification','selected_by_validation_logloss','logloss'),('regression','selected_by_validation_mae','mae_log10')]:
        for task,models in cm[category].items():
            assert models[selection]==min((k for k in models if isinstance(models[k],dict)),key=lambda k:models[k]['validation'][score])
    pd.DataFrame(records).to_csv(out/'ml_metrics.csv',index=False)
    # Unified inference replay on all1000, plus unsupported profiles.
    prototype=QualityPrototype(out/'registry.json'); answers=[]
    with np.load(DATA/'all_methods_equations_v1_ml.npz',allow_pickle=False) as z:
        first=inputs['metadata'][str(inputs['ids'][0])]
        metadata={key:first[key] for key in CERT_FIELDS}
        metadata['serialized_ast']={key:first['serialized_ast'][key] for key in ('a_raw','b_raw','c_raw')}
        write(out/'input_example.json',dict(domain='all_methods_v1',metadata=metadata,
            coefficients=z['X'][0].tolist(),x_grid=z['x_grid'].tolist()))
        for i,pid in enumerate(inputs['ids']):
            answer=prototype.predict(z['X'][i],z['x_grid'],inputs['metadata'][str(pid)],
                planned_workers=int(rt.iloc[i].planned_external_workers))
            rows={r['profile_id']:r for r in answer['results']}
            check_close(rows['remizov_taper_grid33']['predictions']['p_accuracy_met'],rp.iloc[i].p_success_catboost)
            check_close(rows['remizov_taper_grid33']['predictions']['historical_solver_seconds'],rp.iloc[i].seconds_dummy_context)
            check_close(rows['remizov_taper_x0']['predictions']['p_accuracy_met'],cp.iloc[i].p_remizov_point_accuracy_catboost)
            check_close(rows['mc_x0_single1024']['predictions']['p_accuracy_met'],cp.iloc[i].p_mc_single_seed_accuracy_catboost)
            check_close(rows['remizov_taper_x0']['predictions']['log10_absolute_error'],cp.iloc[i].logerror_remizov_point_log_error_catboost)
            check_close(rows['mc_x0_mean4x1024']['predictions']['log10_absolute_error'],cp.iloc[i].logerror_mc_aggregate_log_error_catboost)
            assert answer['method_selection']['winner'] is None
            answers.append(dict(equation_id=str(pid),**answer))
    with (out/'prototype_predictions.jsonl').open('w') as stream:
        for answer in answers:stream.write(json.dumps(answer,ensure_ascii=False,allow_nan=False)+'\n')
    write(out/'example_prediction.json',answers[0])
    result=dict(models_replayed=models_checked,scalers_train_only_verified=scaler_checks,
        linear_nonblas_checks=linear_independent_checks,
        recomputed_raw_labels=raw_counts,features_identical_rows=1000,feature_count=253,
        forbidden_metadata_invariance_rows=1000,original_split_rows={'train':700,'validation':150,'test':150},
        original_split_families={'train':140,'validation':30,'test':30},family_and_topology_disjoint=True,
        seed_replicas_stay_in_original_split=True,all_saved_metrics_recomputed=True,
        validation_selection_preserved=True,training_sources_match_archives=True,
        unified_predictions_reproduced=1000,unknown_remizov_targets_censored=3,
        new_solver_calls=0,new_model_fits=0,
        limitations=['Source and artifact audit cannot retrospectively prove absence of every hidden training input',
            'Historical test previously exposed; evaluation is descriptive, not pristine prospective policy validation',
            'MC feature_names NPZ entry is object dtype; not unpickled; JSON schema used instead',
            'No per-equation probability confidence bounds, no temporal bias certificate, no OOD validation'])
    write(out/'model_audit.json',result)
    return pd.DataFrame(records)


def tables_and_plots(out, metrics):
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rt=pd.read_csv(R/'targets.csv');eq=pd.read_csv(C/'analysis/equations.csv');att=pd.read_csv(C/'analysis/attempts.csv')
    jt=pd.read_csv(J/'analysis/comparison.csv');ft=pd.read_csv(F/'analysis/comparison.csv')
    js=read(J/'analysis/summary.json');fs=read(F/'analysis/summary.json')
    rows=[]
    def row(experiment,cohort,method,profile,points,units,successes,seconds,caveat):
        rows.append(dict(experiment=experiment,cohort=cohort,method=method,profile=profile,
            points=points,observations=units,observed_pass=successes,solver_seconds=seconds,caveat=caveat))
    row('historical1000','all original splits','R','n128/taper/legacyT',33,997,871,
        float(rt.solver_seconds.sum()),'3 worker exceptions censored; mixed 8/1 external workers')
    row('mc1000_x0','same 997 paired','R','node of historical grid33',1,997,int(eq.remizov_accuracy0.sum()),
        None,'No point-only measured runtime; 33-point time cannot be divided by33')
    row('mc1000_x0','all original splits','MC','1x1024,h1/128,G-T',1,4000,int(att.accuracy_met.sum()),
        float(att.solver_seconds.sum()),'4000 seed outcomes,1000 equations,200 families')
    row('mc1000_x0','all original splits','MC','mean4x1024,h1/128,G-T',1,1000,int(eq.mc_aggregate_accuracy.sum()),
        float(att.solver_seconds.sum()),'Same raw runs as preceding row; do not sum costs twice')
    for cohort in ('representative','stress'):
        sub=jt[jt.cohort==cohort]
        row('grid100',cohort,'R','n128/zero/commonG-T',33,len(sub),int((sub.R_error<=.01).sum()),float(sub.R_seconds.sum()),'train only; 80 representative vs20 targeted, not pooled population rate')
        row('grid100',cohort,'MC','1x1024,h1/128,commonG-T',33,len(sub),int((sub.MC_primary_error<=.01).sum()),float(sub.MC_primary_seconds.sum()),'first seed only; no temporal bias control')
        row('grid100',cohort,'MC','adaptive mean1or4x1024',33,len(sub),int((sub.MC_final_error<=.01).sum()),float(sub.MC_all_solve_seconds.sum()),'12 selected repeats; costs include primary, do not sum again')
    row('final_grid5','five targeted train','R','n128/zero baseline',33,5,int(ft.R128_pass.sum()),float(ft.R128_seconds.sum()),'Copied old results, no new cost; subset of grid100')
    row('final_grid5','five targeted train','R','n256/zero/commonG-T',33,5,int(ft.R256_pass.sum()),float(ft.R256_seconds.sum()),'New direct solves; no population accuracy estimate')
    row('final_grid5','five targeted train','MC','1x4096,h1/128,commonG-T',33,20,int(ft.MC_successes.sum()),float(ft.MC_total_solver_seconds.sum()),'Fresh seed outcomes; no p>=.95 success confirmed')
    row('final_grid5','five targeted train','MC','mean4x4096=16384',33,5,int((ft.MC_mean16384_error<=.01).sum()),float(ft.MC_total_solver_seconds.sum()),'Same raw runs as previous row; different estimator/budget')
    table=pd.DataFrame(rows);table.to_csv(out/'experiments.csv',index=False)
    (out/'plots').mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
    for ax,scope,values in [
        (axes[0],'x=0 (1000 equations)',[('R taper',eq.loc[eq.remizov_known,'remizov_error0']),('MC mean4x1024',eq.mc_aggregate_error)]),
        (axes[1],'Grid33: representative80, train',[('R128 zero',jt.loc[jt.cohort=='representative','R_error']),('MC1024 first seed',jt.loc[jt.cohort=='representative','MC_primary_error'])]),
        (axes[2],'Grid33: targeted5, train',[('R128 baseline',ft.R128_error),('R256 zero',ft.R256_error),('MC mean4x4096',ft.MC_mean16384_error)])]:
        for label,values in values:
            v=np.sort(values.to_numpy());ax.step(np.maximum(v,1e-12),np.arange(1,len(v)+1)/len(v),where='post',label=label)
        ax.axvline(.01,color='black',ls='--',lw=1);ax.set(xscale='log',ylim=(0,1.02),xlabel='Observed absolute / max grid error',ylabel='Empirical CDF',title=scope);ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(out/'plots/observed_quality.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
    tasks=[('historical_taper_grid33','accuracy','Grid33 R taper'),('x0','remizov_point_accuracy','x0 R taper'),('x0','mc_single_seed_accuracy','x0 MC single1024')]
    for ax,(scope,task,title) in zip(axes,tasks):
        sub=metrics[(metrics.scope==scope)&(metrics.task==task)&(metrics.split=='test')]
        ax.bar(sub.model,sub.logloss,color=['#888888' if not k else '#287db3' for k in sub.selected]);ax.set(title=title,ylabel='Test log loss (lower is better)');ax.tick_params(axis='x',rotation=20)
    fig.savefig(out/'plots/ml_logloss.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
    for ax,task,title in zip(axes,('remizov_point_accuracy','mc_single_seed_accuracy'),('R taper x0: 150 equations, 6 failures','MC1024 x0: 600 seed labels, 30 families')):
        calibration=pd.read_csv(C/f'ml/calibration_{task}.csv')
        for model,g in calibration[calibration.split=='test'].groupby('model'):
            ax.plot(g.mean_probability,g.observed_success,'o-',label=model)
        ax.plot([0,1],[0,1],'k--',lw=.8)
        ax.set(xlabel='Mean predicted probability',ylabel='Observed success fraction',title=title)
        ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(out/'plots/calibration.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
    axes[0].scatter(jt.R_seconds,jt.MC_primary_seconds,s=20,c=['#287db3' if v=='representative' else '#d08030' for v in jt.cohort],alpha=.7)
    lim=max(jt.R_seconds.max(),jt.MC_primary_seconds.max())*1.05;axes[0].plot([0,lim],[0,lim],'k--',lw=.7)
    axes[0].set(xlabel='R128 zero solve seconds, grid33',ylabel='MC1024 first seed seconds, grid33',title='Measured cost; MC first seed 0/100 passes')
    names=[p.replace('AMV1_','') for p in ft.problem_id];positions=np.arange(5)
    for shift,values,label in [(-.25,ft.R256_seconds,'R256'),(0,ft.MC_median_seconds,'MC4096 median seed'),(.25,ft.MC_total_solver_seconds,'MC4x4096 total')]:
        axes[1].bar(positions+shift,values,width=.25,label=label)
    axes[1].set(xticks=positions,xticklabels=names,ylabel='Solver seconds, grid33',title='Targeted5; repeats charged in full');axes[1].tick_params(axis='x',rotation=25);axes[1].legend(fontsize=8)
    fig.savefig(out/'plots/measured_cost.png',dpi=170);plt.close(fig)
    return table


def report(out, table):
    def markdown(frame):
        columns=list(frame.columns)
        return '| '+' | '.join(columns)+' |\n| '+' | '.join('---' for _ in columns)+' |\n'+''.join('| '+' | '.join(str(v) for v in row)+' |\n' for row in frame.itertuples(index=False,name=None))
    import pandas as pd
    metric=pd.read_csv(out/'ml_metrics.csv');selected=metric[(metric.split=='test')&metric.selected]
    text='''# Завершение текущего ML-этапа — 2026-10-25

Получен единый **прототип прогнозирования качества фиксированных профилей**.
Новые решения ОДУ/MC, оптимизация параметров и переобучение не выполнялись.
Первоначальная цель — выбор самого быстрого надёжного решателя при E∞≤0.01 —
пока не достигнута. Метод не рекомендуется на основании сравнения несопоставимых
вероятностей, бюджетов и времён. API всегда возвращает `method_selection.status=uncertain`.

## Что умеет прототип

- `remizov_taper_grid33`: вероятность точности исторического R128/Q513,
  dx=.0125,padding12,cubic/taper,legacyT,tail1e-8. Порог риска .07705429175
  зафиксирован прежним validation. Прогноз wall — выбранная по validation
  медиана отдельно для планировавшихся 1/8 внешних workers. Это не CPU-время,
  не counterfactual single-worker оценка и не время достижения точности.
- `remizov_taper_x0`: вероятность .01 и log10 абсолютной ошибки узла x=0
  того же исторического33-точечного решения. Отдельного времени x=0 нет.
- `mc_x0_single1024`: вероятность наблюдаемого успеха одного Numba MC1024
  seed,h1/128,аналитический G-T,tail1e-4. Это не доверительная нижняя граница
  вероятности и не сертификат 95%-надёжности.
- `mc_x0_mean4x1024`: log10 ошибки среднего четырёх независимых seed,
  то есть4096 путей в одной точке. У него нет обученного классификатора успеха
  и модели времени. `10**prediction` — точечная оценка ошибки, не upper bound
  и не условное среднее абсолютной ошибки.
- R cubic/zero n128/n256 и MC grid33 с1024/4096/mean16384 возвращают
  `unsupported_no_trained_model` с пустыми предсказаниями. x0 не переносится
  на33 узла, taper не переносится наzero, бюджет single не заменяется mean4.

Реестр `registry.json` содержит полный scope/параметры, SHA моделей и feature-code.
Он использует только ранее выбранные CatBoost и context-median, без новой
переоценки выбора на test. Совпадение x0 при разных T/профилях само по себе
не делает численные методы сравнимыми по стоимости. Отсутствующие CI прогноза,
OOD-проверка и temporal-bias certificate явно помечены как отсутствующие.

## Аудит утечек, разбиений и воспроизводимости

253 входа восстановлены только из заданных a,b,c,g на исходной256-точечной
сетке, глобальных сертификатов и AST a_raw/b_raw/c_raw. c_R=c+lambda,g_R=-g.
Разности/спектры коэффициентов — признаки, не новые решения уравнений.
g допустим как заданная правая часть, хотя manufactured generator создавал
его из exact. Его связь с решением не является утечкой, если g известен до solve.
Exact, y0/v0, производные решения, y_shape, support/scale, ID/seed/hash,
семейство/split, результаты/time/error не входят в X. planned_workers —
единственный дополнительный предрешательный контекст модели времени.
Признак pre_T вычисляется по прежней legacy-формуле при tail1e-8 и служит
предрешательным дескриптором; он не подменяет фактический сертифицированный
MC-горизонт из конфигурации. Его смысл сохранён ради совместимости весов.

Проверены все1000 feature-строк против обоих сохранённых NPZ; на1000 строках
удаление/порча всей неразрешённой metadata не меняет признаков. NPZ MC содержит
object-dtype feature_names: этот элемент не unpickle-ился, использован JSON schema.
Остальные массивы читаются с allow_pickle=False. Модели joblib локальные,
доверенные и закреплены SHA; загружать произвольные внешние joblib нельзя.

Сохранены700/150/150 уравнений,140/30/30 семейств; пересечений family и topology нет.
Все четыре seed остаются внутри уравнения/семейства.600 MC test-меток — это
150 уравнений из30 семейств, не600 независимых задач.3 R worker_exception
остаются неизвестными целями, не отрицательным классом. Пересчитаны997
исходных R grid/point ошибок и4000 MC seed ошибок; проверены идентичности
AST, источники и хеши. Все21 сохранённых предиктора, включая baseline и
контекстную медиану, повторно загружены; предсказания и метрики пересчитаны.
Для7 scaler-пайплайнов mean/variance/count сверены только с train, включая
повторённые MC seed-строки. Исходники обучения совпали с архивными версиями;
early stopping/threshold/model selection проверены по прежнему validation.
Аудит исходников/артефактов поддерживает отсутствие утечки; это не абсолютное
ретроспективное доказательство каждого внутреннего действия при обучении.

Прежний test уже просматривался в численных аудитах. Поэтому метрики —
описательная оценка сохранённого семейного holdout, не нетронутая prospective
проверка политики выбора solver. В этой задаче test не использовался для
переобучения, смены модели, порогов или признаков.

## ML-метрики

Полный `ml_metrics.csv` содержит baseline,Logistic/Ridge,CatBoost, все splits.
Выбранные ранее модели на test (единицы целей различаются):

'''
    cols=['scope','task','model','roc_auc','logloss','mae_seconds','mae_log10','mae_error']
    text+=markdown(selected.reindex(columns=cols).fillna('—').round(6))
    text+='''
R grid33 CatBoost AUC=.8801, Logistic=.8839, dummy=.5; выбранный порог
даёт balanced accuracy=.8106, на test пропущены2/16 неточных и отклонены34
точных. Разность AUC CB−Logistic family-bootstrap95 [-.0427,.0448] не доказывает
превосходства CB. Runtime context-median MAE=.6767с; CB context=.6519с,
но по validation выбрана медиана; парный95% CI CB−median [-.0906,.0399] включает0.

В x0 CatBoost R AUC=.8947, logloss=.1182 противdummy .1775; accuracy=.96
у обоих, поэтому одна accuracy скрывает разницу вероятностных прогнозов.
Всего6 неуспехов в test. MC single1024 CB AUC=.7350, logloss=.5863 противdummy
.6743 (Logistic .6790); family-bootstrap95 разности CB−dummy logloss
[-.1251,-.0515], условно на наблюдённых seed. Для R соответствующий интервал
[-.1219,-.0184]. Это поддержка качества вероятностных surrogate против
constant baseline, не подтверждение надёжности solver.

MAE log10 error: R .3579 противdummy .4844; MC mean4 .3624 противdummy .4121.
Но raw-error MAE R=1.658 чувствителен к историческим taper-катастрофам;
log-прогноз сглаживает редкие выбросы и не защищает от них. Значимость
преимущества регрессоров отдельно не доказана. Важности сохранены в прежних
отчётах: R-classifier lambda, производные a, вариация g; runtime context CB
81.05% важности planned_workers; важность не означает причинный эффект.
В MC-classifier важнее всего gR_std6.25%,gR_rms5.31%,gR_min4.24%;
в R-point a_d2_rms8.30%,lambda3.59%,b_d1_absmax2.76%. Это ассоциации
в текущем manufactured distribution. Сохранённый список крупнейших test-ошибок
MC — [largest_test_prediction_errors.csv](../mc1000_x0_2026-10-25/ml/largest_test_prediction_errors.csv),
R-grid — [test_error_analysis.csv](../remizov_ml_v1_2026-10-25/test_error_analysis.csv).
На этих ошибках модели не перенастраивались.
В исходном sklearn обучении и новом baseline replay были BLAS RuntimeWarning;
независимый einsum пересчёт всех7 линейных пайплайнов повторил конечные
прогнозы в допуске проверки. Причина предупреждений не установлена;
лог и max-differences сохранены, переобучение не выполнялось.

![Метрики ML](plots/ml_logloss.png)
![Описательная калибровка, прежний test](plots/calibration.png)

## Сопоставление основных завершённых экспериментов

Доли наблюдаемые, не сертификаты. Строки средних используют те же raw runs:
их стоимость нельзя суммировать повторно. Исторический R x0 извлечён из
grid33; время одной точки не измерялось.100 и5 целевых train-случаев
не являются независимым test новых профилей. Стресс20 анализируются отдельно.

'''
    text+=markdown(table[['experiment','cohort','method','profile','points','observations','observed_pass','solver_seconds']].fillna('не измерено').round(3))
    text+='''
Подробные оговорки и единицы: `experiments.csv`. В grid100 дополнительно были
4 boundary-контроля (53.36с) и1 прерванный запрос; всего240 полных результатов,
активный wall3561.57с, solver3474.70с,RSS286.39МиБ. Новые5 дали25/25 ответов,
active628.03с,solver622.81с,RSS281.91МиБ. MC1000:4000/4000,solver1633.25с,
RSS285.06МиБ; частичные guardian-работы учитываются отдельно в исходных
отчётах, а не выдаются за завершённый solve.

![Наблюдаемая точность раздельных областей](plots/observed_quality.png)
![Стоимость при фиксированных параметрах](plots/measured_cost.png)

## Предшествующие проверки и их роль

Это последовательные исследования разных вопросов, не дополнительные
сравнительные ML-наблюдения. Сводный индекс всех сохранённых research-отчётов
и их первых тезисов находится в `research_index.csv`; планы помечены отдельно.

- [Интеграция MC](../monte_carlo_integration_2026-10-24/REPORT.md): EM,
  трапеции, статистика и аналитические тесты; малые проверки не доказывали
  достижение .01 на production.
- [Сертификат G](../forcing_bounds_2026-10-24/survey/REPORT.md): глобальная
  аналитическая граница, а не сеточный максимум; гарантирует хвост, не h-ошибку.
- [M/h/T](../mc_convergence_2026-10-24/REPORT.md): независимые seed;
  уменьшение шума и временное смещение разделены, короткий T экспериментален.
- [Связанный F0002](../mc_f0002_coupled_h_2026-10-24/INTERPRETATION.md):
  4x1024, T63.625,x0; ошибка среднего .03343→.01247 при1/32→1/64,
  общий порядок и практическая надёжность не установлены.
- [Независимый контроль F0002](../mc_f0002_confirmation_2026-10-24/INTERPRETATION.md):
  8x1024 контроль и отдельные8x1024 тройки; обе связанные разности положительны,
  ошибка среднего на1/128 .001554, но2/8 отдельных запусков точны;
  одновременный CI bias [-.01941,.02252] не подтверждает полную .01.
- [Частичный старый pilot](../joint_pilot_2026-10-25/):236 полных
  логических ответов,25 неначатых; не считать выполненным и не смешивать
  с joint_grid100. Он сохранён, не возобновлялся.
- [Numba backend](../mc_optimized_backend_2026-10-25/REPORT.md): эквивалентность
  на одинаковых приращениях; ускорения x0 3.71–21.93x на разных AST,
  подготовка/JIT отдельно; ускорение не исправляет шум и смещение.
- [Boundary](../chernov_boundary_diagnosis_2026-10-24/REPORT.md) и последующие
  train-контроли: исправление taper→zero убрало известные катастрофы
  19704.50→.01872 и444.589→.05357, но само по себе не обеспечило .01.

## Научный итог и решение

**Удалось:** воспроизводимо прогнозировать риск/ошибку отдельных фиксированных
профилей лучше простых констант; реализовать безопасную инфраструктуру,
аналитический хвост и ускоренный эквивалентный MC; показать прямым grid33
сравнением практические ограничения. R256 уменьшил ошибки на всех5,
наблюдаемые успехи2/5→5/5. MC4096 уменьшил stderr примерно вдвое относительно
1024, но дал5/20 точных trial только там, где R уже точен. Отдельные точные
MC-trial быстрее R256; это наблюдение сохраняется, но не доказывает надёжный
выигрыш. Mean16384 всё ещё неточен на3/5, а на2/5 стоит дороже R256.

**Не удалось:** получить уверенные метки самого быстрого надёжного solver.
В x0 лишь67 разделённых exploratory quality-меток (60 R,7 MC; train44/5),
и это не метки скорости. В grid100 и финальной пятёрке надёжные MC-only
случаи не получены; temporal bias не ограничен. Нельзя считать отсутствие
преимущества MC универсальным математическим результатом. Нет модели
cubic/zero grid33 или подтверждённого минимального времени до допуска.

**Текущий ML-этап завершён как prototype качества с abstention.** Не запускать
MC900, не увеличивать M автоматически и не переобучать winner-модель на
однообразных/неопределённых метках. Дальнейшее обучение cubic/zero на нынешних
100 train-случаях не даст независимой validation/test оценки.

Для первоначальной цели в будущем нужны: (1) единые замороженные профили и
прямые стоимости на одной grid33/оборудовании с полным учётом повторов;
(2) конкурентный MC с контролем временного смещения и одновременной
статистической ошибкой, достаточными независимыми seed для объявленной
надёжности; (3) достаточные оба класса воспроизводимых успехов/неуспехов,
включая случаи выгодного MC; (4) семейные calibration/holdout данные
исправленного cubic/zero, а для итоговой проверки политики — новый нетронутый
семейный prospective test. Старые split не менять. Эти требования — перечень
недостающих данных, не разрешение на новые расчёты в текущем этапе.

## Воспроизведение

Из корня проекта, локальная .venv (версии закреплены в environment.json):

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B diagnostics/predict_quality.py \\
  --ids AMV1_F0000_S00 --planned-workers 1
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B diagnostics/predict_quality.py \\
  --ids AMV1_F0047_S00 --profiles remizov_zero_grid33_n256 mc_grid33_single4096
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B diagnostics/close_ml_research.py \\
  --out results/ml_closure_reproduction
```

Python API: `QualityPrototype(registry).predict(values,x_grid,metadata)`;
CLI `--input-json` принимает coefficients[4,256],x_grid,metadata с глобальными
коэффициентными сертификатами и AST a/b/c. Exact/ID/outputs не нужны.
API проверяет схему, но не доказывает глобальные сертификаты произвольного
пользовательского уравнения и не определяет принадлежность training distribution.
Воспроизведение запускает inference/аудит/графики; методов solve/fit в нём нет.
Старые каталоги и веса не изменяются. `model_audit.json` и `verification.json`
содержат проверки; `artifact_manifest.json` закрепляет новые артефакты.
'''
    (out/'REPORT.md').write_text(text)
    section=text.split('## Научный итог и решение\n',1)[1].split('## Воспроизведение',1)[0]
    (out/'INTERPRETATION.md').write_text('# Итог текущего ML-этапа\n\n'+section+
        '\n[Полный отчёт, метрики и воспроизведение](REPORT.md).\n')
    index=[]
    for folder in sorted((ROOT/'results').iterdir()):
        if not folder.is_dir() or folder==out:continue
        for name in ('INTERPRETATION.md','REPORT.md'):
            p=folder/name
            if p.exists():
                lines=[line for line in p.read_text().splitlines() if line.strip() and not line.startswith('#')]
                index.append(dict(directory=folder.name,document=str(p.relative_to(ROOT)),
                    sha256=sha(p),kind='plan_not_result' if 'plan' in folder.name else 'saved_report_check_scope',
                    opening_text=' '.join(lines[:3])[:700]))
    pd.DataFrame(index).to_csv(out/'research_index.csv',index=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    out=parser.parse_args().out.resolve()
    if not out.is_relative_to(ROOT/'results') or out==ROOT/'results':
        parser.error('new results directory required')
    if not out.exists():
        out.mkdir();(out/'sources').mkdir();shutil.copy2(ROOT/'PROJECT_STATUS.md',out/'PROJECT_STATUS.before.md')
        write(out/'protected_before.json',snapshot(out))
        write(out/'closure_protocol.json',dict(schema='ml_research_closure_v1',new_solver_calls=0,new_fits=0))
    elif not (out/'closure_protocol.json').exists() or read(out/'closure_protocol.json').get('schema')!='ml_research_closure_v1':
        parser.error('refusing to overwrite a directory not created for this closure')
    from ml.remizov_features import load_inputs
    inputs=load_inputs(DATA)
    registry(inputs,out)
    print('Registry frozen; replaying saved models, no fit/solve',flush=True)
    metrics=audit(inputs,out)
    table=tables_and_plots(out,metrics)
    report(out,table)
    write(out/'environment.json',dict(python=sys.version,platform=platform.platform(),threads=1,
        versions={k:importlib.metadata.version(k) for k in ('numpy','scipy','pandas','scikit-learn','catboost','joblib','matplotlib')}))
    (out/'requirements-lock.txt').write_text('\n'.join(sorted(
        f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()))+'\n')
    before=read(out/'protected_before.json')
    changed=[p for p,h in before.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    assert not changed,changed
    audit_result=read(out/'model_audit.json')
    write(out/'verification.json',dict(protected_paths=len(before),changed=changed,
        models_replayed=len(audit_result['models_replayed']),new_solver_calls=0,new_model_fits=0,
        unified_predictions=1000,method_selection='uncertain',original_family_split_preserved=True))
    for source in ('ml/quality_prototype.py','diagnostics/predict_quality.py','diagnostics/close_ml_research.py'):
        dest=out/'sources'/source;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/source,dest)
    write(out/'artifact_manifest.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*')
        if p.is_file() and p.name!='artifact_manifest.json'})
    print(json.dumps(read(out/'verification.json'),indent=2),flush=True)


if __name__=='__main__':
    main()
