"""Control-plane regression, without running a numerical method."""
import json
import time
from pathlib import Path

from diagnostics.mc_production_x0 import wait,collect,key


class Child:
    stop_reason=None
    def poll(self):return None


def test_wall_memory_and_global_deadline_are_distinct(tmp_path):
    state=tmp_path/'state.json';stop=tmp_path/'stop.json'
    limits=dict(rss_mib=512,poll_seconds=.001)
    now=time.monotonic()
    state.write_text(json.dumps(dict(phase='solving',monotonic=now,peak_rss_mib=513)))
    assert wait(Child(),tmp_path/'absent',state,now+60,limits,time.time()+14400,stop)[0]=='external_memory_limit'
    state.write_text(json.dumps(dict(phase='solving',monotonic=now,peak_rss_mib=100)))
    assert wait(Child(),tmp_path/'absent',state,now-1,limits,time.time()+14400,stop)[0]=='external_time_limit'
    assert wait(Child(),tmp_path/'absent',state,now+60,limits,time.time()-1,stop)[0]=='series_budget_exhausted'
    stop.touch()
    assert wait(Child(),tmp_path/'absent',state,now+60,limits,time.time()+14400,stop)[0]=='interrupted'


def test_atomic_completed_record_precedes_watchdog_race(tmp_path):
    result=tmp_path/'result.json';result.write_text('{}')
    reason,peak=wait(Child(),result,tmp_path/'absent',0,dict(rss_mib=512),0,tmp_path/'stop')
    assert reason is None


def test_frozen_production_profile_and_resume_census():
    root=Path(__file__).resolve().parents[1];out=root/'results/mc1000_x0_2026-10-25'
    if not (out/'plan.json').exists():return
    p=json.loads((out/'plan.json').read_text())
    assert p['config']['M']==1024 and p['config']['dt']==1/128
    assert p['config']['T'] is None and p['config']['tail_eps']==1e-4
    assert p['target_x']==[0.] and len(set(p['seeds']))==4
    assert len(p['batches'])==50 and all(len(x)==20 for x in p['batches'])
    assert p['resources']['series_seconds']==14400 and p['resources']['attempt_seconds']==60
    done=collect(out,p)
    assert all(k==key(v['problem_id'],v['seed_index']) for k,v in done.items())
