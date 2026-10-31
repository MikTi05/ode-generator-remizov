"""Read-only audit of this design and preserved files; no numerical imports."""
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    before=json.loads((HERE/"protected_before.json").read_text())
    changed=[name for name,digest in before.items()
             if not (ROOT/name).is_file() or sha(ROOT/name)!=digest]
    assert not changed, changed
    proposal=json.loads((HERE/"proposal.json").read_text())
    assert proposal["execution_authorized"] is False
    assert proposal["executable_plan"] is False
    assert proposal["new_solver_runs"] == 0
    rows=list(csv.DictReader((HERE/"resources20.csv").open()))
    assert len(rows)==20
    assert sum(r["grid_in_recommended_pilot"]=="True" for r in rows)==6
    for r in rows:
        T=float(r["T_proposed"]); N=int(r["steps"])
        assert T>=float(r["T_certified_saved"]) and N==128*T
        assert float(r["tail_bound_formula"])<=1e-4
        assert int(r["point_path_steps"])==2048*N
        assert int(r["grid33_path_steps"])==33*1024*N<=300000000
    totals=json.loads((HERE/"resource_summary.json").read_text())
    assert totals["point20_mc_updates_8seeds"]==sum(int(r["point_path_steps"])*8 for r in rows)
    assert totals["grid6_mc_updates_4seeds"]==sum(int(r["grid33_path_steps"])*4 for r in rows
                                                  if r["grid_in_recommended_pilot"]=="True")
    families=list(csv.DictReader((HERE/"family_manifest.csv").open()))
    assert len(families)==1020
    assert Counter(r["split"] for r in families if r["dataset"]=="small20")=={"validation":10,"test":10}
    assert len({(r["dataset_sha256"],r["equation_id"],r["expression_hash"]) for r in families})==1020
    new_seeds={s for phase in proposal["phases"] for s in phase["root_seeds"]}
    old_seeds=set()

    def collect_seeds(obj):
        if isinstance(obj,dict):
            for key,value in obj.items():
                if key in ("seed","seed_entropy","root_seed") and isinstance(value,int):
                    old_seeds.add(value)
                elif key=="root_seeds" and isinstance(value,list):
                    old_seeds.update(x for x in value if isinstance(x,int))
                collect_seeds(value)
        elif isinstance(obj,list):
            for value in obj:
                collect_seeds(value)

    mc_json=0
    for folder in (ROOT/"results").iterdir():
        if folder.is_dir() and (folder.name.startswith("mc_") or folder.name.startswith("monte_carlo_")):
            for path in folder.rglob("*.json"):
                collect_seeds(json.loads(path.read_text()))
                mc_json+=1
    assert not (new_seeds & old_seeds), new_seeds & old_seeds
    broken=[]
    links_checked=0
    for path in [HERE/"PLAN.md",HERE/"RESULTS_AND_LABELS.md",HERE/"RESOURCE_FORECAST.md",ROOT/"PROJECT_STATUS.md"]:
        for target in re.findall(r"\]\(([^)]+)\)",path.read_text()):
            if "://" in target or target.startswith("#"):
                continue
            links_checked+=1
            if not (path.parent/target.split("#")[0]).exists():
                broken.append(dict(document=str(path.relative_to(ROOT)), target=target))
    # PROJECT_STATUS may contain unrelated old links: distinguish them in output.
    new_broken=[b for b in broken if b["document"]!="PROJECT_STATUS.md" or "joint_benchmark_plan" in b["target"]]
    assert not new_broken, new_broken
    previous=(HERE/"PROJECT_STATUS.before.md").read_text()
    current=(ROOT/"PROJECT_STATUS.md").read_text()
    old_body=previous.split("\n\n",1)[1].replace(
        "Обновлено: **2026-10-25**. Завершена ограниченная независимая проверка F0002",
        "Предыдущий этап: завершена ограниченная независимая проверка F0002",1)
    assert current.endswith(old_body)
    evidence=json.loads((HERE/"historical_evidence.json").read_text())
    assert evidence["raw_records"]==1000 and evidence["raw_exact_errors_rechecked"]==997
    report=dict(design_only=True,new_ode_solver_runs=0,new_benchmark_runs=0,new_ml_runs=0,
                protected_files=len(before),protected_files_changed=changed,
                historical_records_read=1000,saved_error_vectors_rechecked=997,
                resource_rows=20,identity_rows=1020,proposed_attempts_max=262,
                mc_old_json_examined=mc_json,proposed_root_seeds=len(new_seeds),
                overlap_with_recorded_mc_seed_fields=sorted(new_seeds & old_seeds),
                local_links_checked=links_checked,new_broken_links=new_broken,
                preexisting_status_broken_links=broken,status_history_preserved=True,
                source_sha256={name:sha(ROOT/name) for name in
                    ("remizov_solver.py","chernov_solver.py","mc_solver.py",
                     "all_methods_bounds.py","all_methods_dataset_loader.py",
                     "src/all_methods_dataset_v1.py","src/all_methods_expressions.py")},
                status_before_sha256=sha(HERE/"PROJECT_STATUS.before.md"),
                status_after_sha256=sha(ROOT/"PROJECT_STATUS.md"),
                pure_planning_tests={"passed":10,"command":"PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_joint_benchmark_plan.py"},
                full_pytest_run=False)
    (HERE/"verification.json").write_text(json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("source_sha256",)},indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
