"""Batch persistence and supervisor tests, independent of a long ODE solve."""
import json
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from diagnostics.run_chernov_batch import (accuracy_status, collect_completed,
    limit_reason, plan_hash, terminate, validate_identity, wait_for_result, write_json)
from diagnostics.report_chernov_batch import distribution


def test_atomic_exclusive_results_preserve_old_bytes(tmp_path):
    p=tmp_path/'result.json'
    write_json(p,{'a':[1.,float('nan')]},exclusive=True)
    previous=p.read_bytes()
    with pytest.raises(FileExistsError):
        write_json(p,{'a':2},exclusive=True)
    assert p.read_bytes()==previous
    assert not list(tmp_path.glob('.batch_*'))
    assert json.loads(previous)=={'a':[1.,None]}


def fixture_plan():
    return dict(equation_ids=['A'],config={'n':128})


def result(plan):
    return dict(problem_id='A',attempt_id='one',config=plan['config'],
        plan_sha256=plan_hash(plan),implementation='chernov_solver.ChernovSolver.solve')


def test_resume_adopts_committed_attempt_and_skips_existing_result(tmp_path):
    plan=fixture_plan();r=result(plan)
    (tmp_path/'solutions').mkdir();(tmp_path/'attempts/one').mkdir(parents=True)
    write_json(tmp_path/'attempts/one/result.json',r,exclusive=True)
    first=collect_completed(tmp_path,plan)
    p=tmp_path/'solutions/A.json';old=p.read_bytes()
    assert first=={'A':r} and collect_completed(tmp_path,plan)==first
    assert p.read_bytes()==old
    with pytest.raises(ValueError):
        collect_completed(tmp_path,dict(plan,config={'n':256}))


def test_resume_leaves_unfinished_request_pending_and_preserved(tmp_path):
    plan=fixture_plan()
    (tmp_path/'solutions').mkdir();(tmp_path/'attempts/one').mkdir(parents=True)
    p=tmp_path/'attempts/one/request.json';write_json(p,{'problem_id':'A'},exclusive=True)
    assert collect_completed(tmp_path,plan)=={}
    assert json.loads(p.read_text())=={'problem_id':'A'}


def test_resume_rejects_two_terminal_attempts_for_one_equation(tmp_path):
    plan=fixture_plan();r=result(plan)
    (tmp_path/'solutions').mkdir();(tmp_path/'attempts/one').mkdir(parents=True)
    (tmp_path/'attempts/two').mkdir(parents=True)
    write_json(tmp_path/'attempts/one/result.json',r,exclusive=True)
    write_json(tmp_path/'attempts/two/result.json',dict(r,attempt_id='two'),exclusive=True)
    with pytest.raises(ValueError,match='multiple terminal'):
        collect_completed(tmp_path,plan)


def test_old_implementation_cannot_be_adopted():
    plan=fixture_plan()
    with pytest.raises(ValueError,match='wrong numerical implementation'):
        validate_identity(dict(result(plan),implementation='remizov_solver.solve_remizov'),plan)


@pytest.mark.parametrize('elapsed,rss,expected',[
    (121,100,'external_time_limit'),(10,513,'external_memory_limit'),(10,100,None)])
def test_limits(elapsed,rss,expected):
    assert limit_reason(elapsed,rss,120,512)==expected


@pytest.mark.parametrize('mode,expected',[
    ('timeout','external_time_limit'),('memory','external_memory_limit'),
    ('interrupt','interrupted'),('crash','worker_exception')])
def test_watchdog_stops_and_reaps_worker(tmp_path,mode,expected):
    state=tmp_path/'state.json'
    write_json(state,dict(phase='solving',peak_rss_mib=600 if mode=='memory' else 1))
    child=subprocess.Popen([sys.executable,'-c',
        'import sys,time; sys.exit(2)' if mode=='crash' else 'import time; time.sleep(10)'])
    stop=threading.Event()
    if mode=='interrupt':stop.set()
    try:
        reason,elapsed,peak=wait_for_result(child,tmp_path/'result.json',state,
            2 if mode=='crash' else .05,512,.01,stop)
        assert reason==expected and child.poll() is not None
    finally:
        terminate(child)


def test_saved_result_is_observed_without_stopping_healthy_worker(tmp_path):
    p=tmp_path/'result.json';write_json(p,{'done':True})
    child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(10)'])
    try:
        assert wait_for_result(child,p,tmp_path/'state.json',1,512,.01,threading.Event())[0] is None
        assert child.poll() is None
    finally:
        terminate(child)


def test_accuracy_and_distribution_do_not_hide_misses():
    assert accuracy_status('success',.01,.01)=='accuracy_pass'
    assert accuracy_status('success',.0101,.01)=='accuracy_not_met'
    assert accuracy_status('external_time_limit',None,.01)=='external_time_limit'
    assert distribution([1,2,3])['median']==2
    assert distribution([])=={'count':0}
