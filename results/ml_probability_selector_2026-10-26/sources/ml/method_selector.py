"""Input-only CatBoost selector helpers and a fail-closed serving API.

Diagnostic train-only models are saved for research, never deployed as complete
R/MC selectors. Unknown is censored evidence, not a fourth learned class.
"""
from __future__ import annotations
import hashlib,json,time
from pathlib import Path
import numpy as np
from ml.selector_labels import LABELS, LABEL_VERSION
from ml.remizov_features import coefficient_features, VERSION, CERT_FIELDS


def classification_metrics(truth,predictions):
    from sklearn.metrics import confusion_matrix,f1_score
    y=np.asarray(truth);p=np.asarray(predictions)
    if y.ndim!=1 or p.ndim!=1 or len(y)!=len(p) or not set(y)<=set(LABELS) or not set(p)<=set(LABELS):raise ValueError('known labels only')
    counts={c:int(np.sum(y==c)) for c in LABELS};present=[c for c in LABELS if counts[c]]
    return dict(known=len(y),accuracy=float(np.mean(y==p)) if len(y) else None,
        macro_f1_supported=float(f1_score(y,p,labels=present,average='macro',zero_division=0)) if len(y) else None,
        support=counts,recall={c:float(np.mean(p[y==c]==c)) if counts[c] else None for c in LABELS},
        confusion_labels=list(LABELS),confusion=confusion_matrix(y,p,labels=LABELS).tolist() if len(y) else [[0]*3 for _ in LABELS])


def family_bootstrap(y,p,groups,seed=2488,repeats=500):
    y=np.asarray(y);p=np.asarray(p);groups=np.asarray(groups);unique=np.unique(groups)
    if not len(unique):return dict(status='no_evidence')
    rng=np.random.default_rng(seed);values=[]
    indices={g:np.flatnonzero(groups==g) for g in unique}
    for _ in range(repeats):
        sample=np.concatenate([indices[g] for g in rng.choice(unique,size=len(unique),replace=True)])
        values.append(float(np.mean(y[sample]==p[sample])))
    return dict(confidence=.95,families=len(unique),resamples=repeats,seed=seed,
        accuracy_percentile95=np.quantile(values,[.025,.975]).tolist(),
        scope='whole-family resampling of fixed OOF predictions; exploratory, conditional on fitted folds')


def fit_catboost(X,y,names,*,validation=None,iterations=200,seed=2488):
    from sklearn.preprocessing import StandardScaler
    from catboost import CatBoostClassifier,Pool
    y=np.asarray(y)
    if len(set(y))<2 or not set(y)<=set(LABELS):raise ValueError('at least two observed classes; unknown forbidden')
    scaler=StandardScaler().fit(X)
    model=CatBoostClassifier(iterations=iterations,depth=3,learning_rate=.05,l2_leaf_reg=5.,
        loss_function='MultiClass',random_seed=seed,thread_count=1,allow_writing_files=False,verbose=False,
        bootstrap_type='No')
    kwargs={}
    if validation is not None:
        vx,vy=validation
        if not set(vy)<=set(y):raise ValueError('validation contains an untrained class')
        kwargs=dict(eval_set=Pool(scaler.transform(vx),label=vy,feature_names=names),early_stopping_rounds=30,use_best_model=True)
    started=time.perf_counter_ns()
    model.fit(Pool(scaler.transform(X),label=y,feature_names=names),**kwargs)
    return scaler,model,(time.perf_counter_ns()-started)*1e-9


def predict_model(scaler,model,X,names):
    from catboost import Pool
    began=time.perf_counter_ns();p=model.predict_proba(Pool(scaler.transform(X),feature_names=names),thread_count=1)
    elapsed=(time.perf_counter_ns()-began)*1e-9
    return np.asarray(model.classes_)[np.argmax(p,axis=1)],p,elapsed


def train_only_diagnostic(X,rows,names,*,seed=2488,iterations=200):
    """A held-out family never participates in its scaler/model fitting."""
    from sklearn.model_selection import StratifiedGroupKFold
    idx=np.asarray([i for i,r in enumerate(rows) if r['split']=='train' and r['label']!='unknown'],dtype=int)
    y=np.asarray([rows[i]['label'] for i in idx]);groups=np.asarray([rows[i]['family_id'] for i in idx])
    counts={c:len(set(groups[y==c])) for c in set(y)}
    if len(counts)<2 or min(counts.values())<2:
        return dict(status='insufficient_diagnostic_classes',fit_indices=idx.tolist(),model=None)
    folds=min(4,min(counts.values()));cv=StratifiedGroupKFold(n_splits=folds,shuffle=True,random_state=seed)
    pred=np.empty(len(idx),dtype=object);fold_audit=[];oof_rows=[]
    for fold,(train,hold) in enumerate(cv.split(X[idx],y,groups)):
        if set(groups[train])&set(groups[hold]):raise ValueError('CV family leakage')
        scaler,model,seconds=fit_catboost(X[idx[train]],y[train],names,iterations=iterations,seed=seed)
        yp,proba,inference=predict_model(scaler,model,X[idx[hold]],names);pred[hold]=yp
        fold_audit.append(dict(fold=fold,fit_indices=idx[train].tolist(),held_out_indices=idx[hold].tolist(),
            fit_families=sorted(set(groups[train])),held_out_families=sorted(set(groups[hold])),
            scaler_mean=scaler.mean_.tolist(),fit_seconds=seconds,inference_seconds=inference))
        for j,pos in enumerate(hold):
            i=int(idx[pos]);oof_rows.append(dict(index=i,equation_id=rows[i]['equation_id'],family_id=rows[i]['family_id'],split='train',fold=fold,truth=y[pos],prediction=str(yp[j]),probabilities={str(c):float(proba[j,k]) for k,c in enumerate(model.classes_)}))
    metrics=classification_metrics(y,pred);constant=classification_metrics(y,['remizov']*len(y))
    scaler,model,seconds=fit_catboost(X[idx],y,names,iterations=iterations,seed=seed)
    return dict(status='diagnostic_only',fit_indices=idx.tolist(),classes=list(model.classes_),model=model,scaler=scaler,
        metrics=metrics,constant_remizov=constant,bootstrap=family_bootstrap(y,pred,groups,seed),
        baseline_bootstrap=family_bootstrap(y,np.array(['remizov']*len(y)),groups,seed),
        oof=oof_rows,folds=fold_audit,final_fit_seconds=seconds,
        scope='R_vs_none_train_only' if set(y)=={'remizov','none'} else 'low_support_train_only',
        original_validation_or_test_used=False,MC_recognition_learned='monte_carlo' in set(y))


def retrospective_policy(rows,evidence,choices):
    """No counterfactual missing outputs or legacy times imputed as full costs."""
    if len(choices)!=len(rows) or not set(choices)<=set(LABELS):raise ValueError('one valid action per requested row required')
    known=[i for i,r in enumerate(rows) if r['label']!='unknown']
    truth=[rows[i]['label'] for i in known];pred=[choices[i] for i in known]
    selected=[];unobserved=0;abstain=0;correct=0;violations=0;failures=0
    for row,choice in zip(rows,choices):
        if choice=='none':abstain+=1;continue
        e=evidence[row['equation_id']+':'+choice]
        if e['status']=='not_run':unobserved+=1;continue
        if e['status']!='completed':failures+=1;continue
        selected.append(e);ok=e['error']<=.01;correct+=ok;violations+=not ok
    return dict(classification=classification_metrics(truth,pred),requested_tasks=len(rows),known_labels=len(known),
        known_label_fraction=len(known)/len(rows) if rows else None,
        observed_finite_outputs=len(selected),unobserved_selected_outputs=unobserved,
        observed_correct_deliveries=correct,observed_accuracy_violations=violations,observed_execution_failures=failures,
        conditional_accuracy_on_observed_outputs=correct/len(selected) if selected else None,
        delivery_accuracy_rate=correct/len(rows) if rows and not unobserved else None,
        abstention_rate=abstain/len(rows) if rows else None,total_policy_wall_seconds=None,
        cost_reason='no common complete end-to-end costs, no actual policy/solver execution',
        accuracy_constrained_cost_regret=None,regret_reason='no uniform median4 R cost evidence',
        evaluation='retrospective saved outputs only; missing selected output is unobserved, not a fabricated failure')


class MethodSelector:
    """Serve only a byte-pinned full selector; diagnostics always abstain."""
    def __init__(self,registry):
        self.path=Path(registry);self.registry=json.loads(self.path.read_text());self.model=None;self.scaler=None
        r=self.registry
        if r['label_type']!=LABEL_VERSION or r['feature_version']!=VERSION:raise ValueError('schema mismatch')
        source=Path(__file__).with_name('remizov_features.py')
        if hashlib.sha256(source.read_bytes()).hexdigest()!=r['feature_code_sha256']:raise ValueError('feature code mismatch')
        if len(r['feature_names'])!=253 or len(set(r['feature_names']))!=253:raise ValueError('feature count/order mismatch')
        if r['selector_status']=='trained':
            if not {'remizov','monte_carlo'}<=set(r['classes']):raise ValueError('incomplete method classes cannot be deployed')
            from catboost import CatBoostClassifier
            for key in ('model','scaler'):
                a=r[key];p=self.path.parent/a['path']
                if hashlib.sha256(p.read_bytes()).hexdigest()!=a['sha256']:raise ValueError('model/scaler integrity mismatch')
            self.model=CatBoostClassifier();self.model.load_model(str(self.path.parent/r['model']['path']))
            from sklearn.preprocessing import StandardScaler
            self.scaler=StandardScaler()
            with np.load(self.path.parent/r['scaler']['path'],allow_pickle=False) as z:
                self.scaler.mean_=z['mean'];self.scaler.scale_=z['scale'];self.scaler.var_=z['var'];self.scaler.n_features_in_=253
            if list(self.model.feature_names_)!=r['feature_names'] or list(self.model.classes_)!=r['classes']:raise ValueError('model schema mismatch')

    def select_method(self,coefficients,x_grid,coefficient_metadata,*,profile_pair_id,timing_context,domain='all_methods_v1'):
        r=self.registry
        result=dict(schema='method_selector_v1',label_type=LABEL_VERSION,profile_pair_id=profile_pair_id,
            action='abstain',predicted_label='unknown',status='insufficient_evidence',probabilities=None,
            predicted_warm_cost_seconds=None,reliability_status='not_certified',accuracy_certificate=False)
        if profile_pair_id!=r['profile_pair_id'] or timing_context!=r['timing_context']:
            return dict(result,status='unsupported_profile',reason='profile_or_timing_context_mismatch')
        x=np.asarray(x_grid,dtype=float)
        if domain!='all_methods_v1' or x.shape!=(256,) or not np.isfinite(x).all() or hashlib.sha256(x.tobytes()).hexdigest()!=r['input_grid_sha256']:
            return dict(result,status='out_of_domain',reason='domain_or_input_grid_mismatch')
        try:
            c={k:float(coefficient_metadata[k]) for k in CERT_FIELDS}
            if not all(np.isfinite(v) for v in c.values()) or not 0<c['a_lower_cert']<=c['a_upper_cert'] or c['b_abs_cert']<0 or c['remizov_c_abs_cert']<0 or c['remizov_lambda']<=c['remizov_c_abs_cert']:
                raise ValueError('invalid coefficient certificates')
            f=coefficient_features(coefficients,x,coefficient_metadata)
            if list(f)!=r['feature_names']:raise ValueError('feature schema mismatch')
        except (ValueError,KeyError,TypeError,OverflowError):return dict(result,status='out_of_domain',reason='invalid_input_coefficients_or_certificates')
        if self.model is None:return dict(result,reason='full_selector_not_trained;'+','.join(r['blocking_reasons']))
        y,p,_=predict_model(self.scaler,self.model,np.asarray([[f[n] for n in r['feature_names']]]),r['feature_names']);label=str(y[0])
        return dict(result,predicted_label=label,action='abstain' if label=='none' else label,status='prediction',
            probabilities={str(c):float(p[0,i]) for i,c in enumerate(self.model.classes_)},reason='observed_label_prediction_not_accuracy_certificate')
