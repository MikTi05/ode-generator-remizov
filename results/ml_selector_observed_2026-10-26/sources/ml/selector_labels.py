"""Versioned observed-fastest labels. No solver calls, model fitting or I/O.

Eligibility is episode0; repetitions are never independent training examples.
Runtime is compared only when both methods are accurate (CONTRACT sections4–6).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import json
import math
import statistics

LABEL_VERSION = 'observed_fastest_v1'
LABELS = ('remizov', 'monte_carlo', 'none')
ALL_LABELS = (*LABELS, 'unknown')
TIMING_SCOPE = 'warm_end_to_end_method_wall_v1'
GRID = tuple(-1+i/16 for i in range(33))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


@dataclass
class Evidence:
    method: str
    equation_id: str
    expression_hash: str
    family_id: str
    split: str
    profile: str
    status: str = 'not_run'
    error: float | None = None
    T: float | None = None
    times: list[float] = field(default_factory=list)
    timing_scope: str | None = None
    timing_cohort: str | None = None
    source: list[str] = field(default_factory=list)
    source_sha256: list[str] = field(default_factory=list)
    records: list[dict] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    deterministic_repeats_equal: bool = False

    @property
    def eligibility(self):
        if self.status!='completed' or self.error is None or not math.isfinite(self.error) or self.error<0:
            return 'unknown'
        return 'pass' if self.error<=.01 else 'fail'


def timing_check(r: Evidence, mc: Evidence):
    reasons=[]
    medians={}
    for e in (r,mc):
        if e.timing_scope!=TIMING_SCOPE:reasons.append(e.method+':wrong_timer_scope')
        if not e.timing_cohort:reasons.append(e.method+':unknown_timing_cohort')
        if len(e.times)!=4 or any(not math.isfinite(t) or t<=0 for t in e.times):
            reasons.append(e.method+':four_uncensored_times_required');continue
        med=statistics.median(e.times);medians[e.method]=med
        if statistics.median(abs(t-med) for t in e.times)/med>.20:
            reasons.append(e.method+':timing_unstable')
    if r.timing_cohort!=mc.timing_cohort:reasons.append('timing_cohort_mismatch')
    if not r.deterministic_repeats_equal:reasons.append('remizov:repeat_reproducibility_unverified')
    return dict(comparable=not reasons,reasons=reasons,medians=medians)


def label_pair(r: Evidence, mc: Evidence, pair_id: str):
    if (r.method,mc.method)!=('remizov','monte_carlo'):raise ValueError('ordered R/MC evidence required')
    row=dict(equation_id=r.equation_id,family_id=r.family_id,split=r.split,expression_hash=r.expression_hash,
        profile_pair_id=pair_id,label_type=LABEL_VERSION,episode=0,label='unknown',tie=False,
        reason='unknown_numerical_evidence',admissible=[],r_eligibility=r.eligibility,mc_eligibility=mc.eligibility,
        r_error=r.error,mc_error=mc.error,time_comparison_required=False,oracle_min_seconds=None,
        r_profile=r.profile,mc_profile=mc.profile)
    if any(getattr(r,k)!=getattr(mc,k) for k in ('equation_id','expression_hash','family_id','split')):
        row['reason']='identity_or_split_mismatch';return row
    if 'unknown' in (r.eligibility,mc.eligibility):return row
    if r.T is None or mc.T is None or r.T!=mc.T:
        row['reason']='common_horizon_mismatch';return row
    row['admissible']=[e.method for e in (r,mc) if e.eligibility=='pass']
    timing=timing_check(r,mc);row['timing']=timing
    if not row['admissible']:row.update(label='none',reason='both_episode0_inaccurate');return row
    if len(row['admissible'])==1:
        row.update(label=row['admissible'][0],reason='sole_episode0_accurate');return row
    row['time_comparison_required']=True
    if not timing['comparable']:row['reason']='timing_unknown';return row
    tr=timing['medians']['remizov'];tm=timing['medians']['monte_carlo']
    row['oracle_min_seconds']=min(tr,tm)
    if tr<=.90*tm and max(r.times)<min(mc.times):row.update(label='remizov',reason='measured_faster')
    elif tm<=.90*tr and max(mc.times)<min(r.times):row.update(label='monte_carlo',reason='measured_faster')
    else:row.update(label='remizov',tie=True,reason='practical_tie_prefer_remizov')
    return row


def support(rows):
    result={}
    for split in ('train','validation','test'):
        own=[r for r in rows if r['split']==split]
        result[split]=dict(total=len(own),known=sum(r['label']!='unknown' for r in own),
            counts={c:sum(r['label']==c for r in own) for c in ALL_LABELS},
            families={c:len({r['family_id'] for r in own if r['label']==c}) for c in ALL_LABELS})
    return result


def training_gate(rows):
    """Predeclared MC-recognition support gate; never synthesize absent classes."""
    s=support(rows);reasons=[]
    if s['train']['families']['monte_carlo']<20:reasons.append('need_20_train_MC_winner_families')
    if s['validation']['families']['monte_carlo']<5:reasons.append('need_5_validation_MC_winner_families')
    if not s['train']['counts']['remizov']:reasons.append('missing_train_remizov_class')
    if not s['validation']['known']:reasons.append('no_original_validation_labels')
    train_classes={r['label'] for r in rows if r['split']=='train' and r['label']!='unknown'}
    val_classes={r['label'] for r in rows if r['split']=='validation' and r['label']!='unknown'}
    if val_classes-train_classes:reasons.append('validation_class_missing_in_train')
    return dict(allowed=not reasons,reasons=reasons,support=s,
        test_required_for_performance_claim=True,test_available=bool(s['test']['known']))
