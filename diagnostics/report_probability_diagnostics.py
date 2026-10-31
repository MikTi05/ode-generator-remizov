"""Saved-only uncertainty/calibration summaries and an unexecuted validation design."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from diagnostics.train_method_selector import write
from ml.selector_evidence import read,sha,CONTRACT
from ml.probabilistic_selector import choose


def report(out):
    import numpy as np
    cv=read(out/'family_cv.json');rows=cv['oof'];rng=np.random.default_rng(2488);metric={}
    for profile in ('R128','R256','MC1'):
        y=np.array([r['observed'][profile]['k']/r['observed'][profile]['n'] for r in rows])
        pred=np.array([r['estimates'][profile]['probability']['point'] for r in rows]);base=np.array([r['baseline_predictions'][profile+':probability'] for r in rows])
        loss=y*(1-pred)**2+(1-y)*pred**2;bloss=y*(1-base)**2+(1-y)*base**2
        draws=[float(np.mean((loss-bloss)[rng.integers(0,len(rows),size=len(rows))])) for _ in range(1000)]
        bins=[];edges=[0,.05,.1,.25,.5,.75,1.00000001]
        for lo,hi in zip(edges,edges[1:]):
            own=[r for r,v in zip(rows,pred) if lo<=v<hi]
            if not own:continue
            k=sum(r['observed'][profile]['k'] for r in own);n=sum(r['observed'][profile]['n'] for r in own)
            bins.append(dict(prediction_bin=[lo,min(hi,1.)],families=len(own),successes=k,trials=n,rate=k/n,mean_prediction=float(np.mean([r['estimates'][profile]['probability']['point'] for r in own]))))
        metric[profile]=dict(Brier_delta_model_minus_constant=float(np.mean(loss-bloss)),conditional_family_bootstrap95=np.quantile(draws,[.025,.975]).tolist(),calibration_bins=bins)
    write(out/'metric_uncertainty.json',dict(seed=2488,resamples=1000,families=40,metrics=metric,scope='whole-family resampling of fixed train OOF predictions; exploratory conditional intervals, not prospective validation or per-input p bounds'))
    a=read(out/'seen_family_fresh_seed_audit.json');est=a['estimates'];illustration={}
    for q in (.5,.8,.95):
        pair={k:v for k,v in est.items() if k in ('R256','MC1')}
        illustration[str(q)]=dict(all_profiles_point=choose(est,q),fixed_pair_point=choose(pair,q),fixed_pair_model_spread=choose(pair,q,robust=True))
    write(out/'seen_family_policy_illustration.json',dict(case=a['equation_id'],choices=illustration,fresh_MC_k=a['fresh_k'],fresh_MC_n=a['fresh_n'],status='in-sample feature case, post-fit seed diagnostic; not evidence of model generalization'))
    v=read(out/'NEXT_VALIDATION.json');contract=read(CONTRACT/'plan.json')
    write(out/'VALIDATION_DESIGN.json',dict(execution_authorized=False,cases=v['cases'],frozen_registry_sha256=sha(out/'registry.json'),
        profiles={k:contract['profiles'][k] for k in ('R128','R256','MC1')},main_calls=64,full_three_profile_calls=96,
        proposed_MC_seed_group=[248800060100+i for i in range(4)],seed_semantics='Same4 base seeds across IDs; point_address_v1 includes equation identity, so streams differ by ID; verify nonreuse before any execution',
        fresh_primary='episode0 fixed; all4 seeds scored separately; no winner hunting',
        timing='one worker/thread, same cohort; full method wall; shared JIT/load separate; all4 uncensored times',
        preflight_required=['canonical hashes/identities/splits','source/profile equality','analytic common G/T and actual work','new seed nonreuse','saved-output scan','resource and total-cost budget from saved scenarios'],
        purpose='held-family diagnostic of frozen p/time heads and point/uncertainty policy; not complete calibration or per-instance reliability certification',
        calibration='Do not fit a calibrator and claim its evaluation on these same8 cases; separate calibration/evaluation design before expansion',
        thresholds=[.5,.8,.95],test_outcomes_used=False,benchmark1000=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);report(ap.parse_args().out.resolve())
