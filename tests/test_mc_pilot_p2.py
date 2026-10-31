import json
from diagnostics import mc_pilot_p2 as p2

def inputs():
    c=p2.read_json(p2.CONTRACT/'plan.json')
    j=[json.loads(s) for s in (p2.CONTRACT/'pilot_jobs.jsonl').read_text().splitlines()]
    return c,j

def test_frozen_p2_scope_and_seed_disjointness():
    c,j=inputs();q=p2.scope_jobs(c,j)
    assert len(q)==140 and len({x['equation_id'] for x in q})==35
    assert not set(c['pilot']['sentinels'])&{x['equation_id'] for x in q}
    assert {x['config']['seed'] for x in q}==set(range(248800020100,248800020104))
    assert not {x['config']['seed'] for x in q}&{x['config']['seed'] for x in j if x['stage']==1}
    assert p2.scope_jobs(c,list(reversed(j)))==q

def test_reject_wrong_seed_or_parameters():
    import pytest
    for field,value in [('seed',1),('M',1024),('dt',1/256)]:
        c,j=inputs();q=next(x for x in j if x['stage']==2 and x['profile']=='MC1')
        q['config'][field]=value
        with pytest.raises(AssertionError):p2.scope_jobs(c,j)

def test_worker_and_timer_unchanged_from_p1():
    import inspect
    from diagnostics import mc_sentinel_p1 as p1
    assert inspect.getsource(p2.worker)==inspect.getsource(p1.worker)
    assert inspect.getsource(p2.measured_call)==inspect.getsource(p1.measured_call)
    assert inspect.getsource(p2.validate_result)==inspect.getsource(p1.validate_result)
