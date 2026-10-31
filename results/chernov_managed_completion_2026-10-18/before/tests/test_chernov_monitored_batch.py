"""Telemetry gates and bounded selection; no real ODE calculations."""
from types import SimpleNamespace

import pytest

import diagnostics.chernov_monitored_batch as monitored


def test_activity_parser_counts_compute_worker_not_guardian_or_controller():
    root=str(monitored.ROOT)
    raw=f'''10 1 10 0.1 3000 python -B {root}/diagnostics/run_chernov_batch.py --output-dir results/x
11 10 11 0.1 4000 python -B {root}/diagnostics/chernov_batch_control.py --command-json ["{root}/diagnostics/run_chernov_batch.py", "--worker-plan", "x"]
12 11 12 99.5 102400 python -B {root}/diagnostics/run_chernov_batch.py --worker-plan x
'''
    assert monitored.parse_activity(raw)==[dict(pid=12,ppid=11,pgid=12,cpu_percent=99.5,rss_mib=100.)]


@pytest.mark.parametrize('level,workers,expected',[
    (0,[],None),(1,[],'thermal_state_not_nominal'),
    (2,[],'thermal_state_not_nominal'),(3,[],'thermal_state_not_nominal'),
    (0,[dict(cpu_percent=100,rss_mib=180)],None),
    (0,[dict(cpu_percent=100,rss_mib=180)]*2,'multiple_compute_workers'),
    (0,[dict(cpu_percent=126,rss_mib=180)],'worker_cpu_above_single_core_budget'),
    (0,[dict(cpu_percent=90,rss_mib=513)],'worker_memory_limit')])
def test_fail_closed_gates(level,workers,expected):
    assert monitored.safety_reason(dict(thermal_state=level),workers)==expected


def test_cooldown_includes_unfinished_monitor_owner_lease(tmp_path):
    assert monitored.cooldown_remaining(tmp_path,1000)==0
    monitored.atomic(tmp_path/'run.json',dict(started_epoch=900,finished_epoch=970))
    assert monitored.cooldown_remaining(tmp_path,1000)==30
    monitored.atomic(tmp_path/'run.json',dict(started_epoch=900,last_sample_epoch=980))
    assert monitored.cooldown_remaining(tmp_path,1000)==160


def test_thermal_failure_does_not_mean_nominal(monkeypatch):
    monkeypatch.setattr(monitored.subprocess,'run',lambda *a,**k:
        SimpleNamespace(returncode=1,stdout='',stderr='unavailable'))
    with pytest.raises(RuntimeError,match='unavailable'):monitored.thermal_sample()


def test_thermal_reading_is_explicitly_not_degrees(monkeypatch):
    monkeypatch.setattr(monitored.subprocess,'run',lambda *a,**k:
        SimpleNamespace(returncode=0,stdout='{"thermalState":0}',stderr=''))
    value=monitored.thermal_sample()
    assert value['temperature_celsius'] is None and value['thermal_name']=='nominal'


def test_no_run_without_acknowledging_measurement_limit(tmp_path):
    with pytest.raises(ValueError,match='Celsius unavailable'):
        monitored.run(tmp_path,tmp_path,3,False)
    with pytest.raises(ValueError,match='1..3'):
        monitored.run(tmp_path,tmp_path,306,True)
