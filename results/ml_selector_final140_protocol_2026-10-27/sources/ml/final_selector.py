"""Finished research inference; estimates feasibility, never certifies actual error.

This is a separate research API. The existing production API stays blocked.
"""
from __future__ import annotations
import hashlib,json,math,time
from pathlib import Path
import numpy as np
from ml.remizov_features import coefficient_features,CERT_FIELDS,VERSION
from ml.revision_selector import ResearchRevision,action,PROFILES


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FinalResearchSelector:
    def __init__(self,model_dir):
        self.folder=Path(model_dir);manifest=json.loads((self.folder/'artifact_sha256.json').read_text())
        for n,h in manifest.items():
            p=(self.folder/n).resolve()
            if self.folder.resolve() not in p.parents or sha(p)!=h:raise ValueError('model artifact integrity failure: '+n)
        self.model=ResearchRevision(self.folder);self.registry=r=self.model.registry
        if r.get('release_schema')!='final_research_selector_v1' or r['feature_version']!=VERSION:raise ValueError('unsupported research release')
        if r['inference_code_sha256']!=sha(__file__):raise ValueError('inference implementation changed')
        if r['engine_code_sha256']!=sha(Path(__file__).with_name('revision_selector.py')):raise ValueError('model engine changed')
        if r['feature_code_sha256']!=sha(Path(__file__).with_name('remizov_features.py')):raise ValueError('feature code changed')
        if r['deployment_ready'] or not r['production_blocked']:raise ValueError('production release not authorized')
        if r['final_policy']['rule']!='parity' or r['final_policy']['mode']!='binary_sigmoid':raise ValueError('unexpected frozen final policy')

    def predict(self,coefficients,x_grid,coefficient_metadata,*,domain='all_methods_v1',profile_context=None,timing_context=None):
        """Select a research profile from inputs only; no solver or result lookup.

        E_grid<=.01 defines the learned success event. Eligibility is the
        train-selected near-best-probability rule, not a numerical certificate
        or a promise of p_new>=.95. Missing context means the frozen context.
        """
        r=self.registry;base=dict(schema=r['release_schema'],status='research_prediction',research_only=True,
            production_allowed=False,accuracy_certificate=False,reliability_certified=False,selected_profile=None)
        if domain!='all_methods_v1':return dict(base,status='unsupported_domain',action='abstain')
        if profile_context is not None and profile_context!=r['profile_context']:return dict(base,status='unsupported_profiles',action='abstain')
        if timing_context is not None and timing_context!=r['timing_context']:return dict(base,status='unsupported_timing_context',action='abstain')
        started=time.perf_counter_ns();x=np.asarray(x_grid,float);values=np.asarray(coefficients,float)
        if x.shape!=(256,) or not np.isfinite(x).all() or sha_bytes(x.tobytes())!=r['input_grid_sha256']:raise ValueError('frozen256 input grid required')
        if values.shape!=(4,256) or not np.isfinite(values).all():raise ValueError('finite a,b,c,g coefficient channels required')
        # Extra metadata (including ID/exact/error) is never inspected or used.
        meta={k:float(coefficient_metadata[k]) for k in CERT_FIELDS}
        if not all(math.isfinite(v) for v in meta.values()) or not 0<meta['a_lower_cert']<=meta['a_upper_cert'] or min(meta['b_abs_cert'],meta['remizov_c_abs_cert'])<0 or meta['remizov_lambda']<=meta['remizov_c_abs_cert']:raise ValueError('invalid analytic coefficient certificates')
        meta['serialized_ast']={k:coefficient_metadata['serialized_ast'][k] for k in ('a_raw','b_raw','c_raw')}
        features=coefficient_features(values,x,meta)
        if list(features)!=r['feature_names']:raise ValueError('feature schema mismatch')
        X=np.array([[features[n] for n in r['feature_names']]])
        variants=self.model.predict_features(X)[0];estimates=variants[r['final_policy']['mode']];profile=action(estimates,r['final_policy'])
        maximum=max(e['p'] for e in estimates.values());eligible=[p for p in PROFILES if estimates[p]['p']>=maximum-r['final_policy']['delta']]
        if profile!='abstain' and profile not in eligible:raise AssertionError('ineligible profile selected')
        return dict(base,action='select_profile' if profile!='abstain' else 'abstain',selected_profile=None if profile=='abstain' else profile,
            selected_method=None if profile=='abstain' else ('monte_carlo' if profile=='MC1' else 'remizov'),
            accuracy_event=dict(metric='max_abs_error_on_fixed_grid33[-1,1]',threshold=.01),
            estimates={p:dict(probability_accuracy=e['p'],predicted_full_warm_seconds=e['seconds'],time_residual_band=[e['time_low'],e['time_high']]) for p,e in estimates.items()},
            predicted_admissible_profiles=eligible,policy=r['final_policy'],
            admissibility_scope='train-selected probability parity; actual E_grid unknown before solve; not p>=.95 certification',
            time_tie_rule='MC requires >=10% predicted saving versus an eligible R profile',
            uncertainty='family OOF/model residual uncertainty, not calibrated per-input reliability bounds',
            timing_context=r['timing_context'],inference_seconds=(time.perf_counter_ns()-started)*1e-9)

    def production_select_method(self,*args,**kwargs):
        return dict(action='abstain',status='production_blocked_pending_independent_evaluation',selected_profile=None,accuracy_certificate=False)


def sha_bytes(value):return hashlib.sha256(value).hexdigest()
