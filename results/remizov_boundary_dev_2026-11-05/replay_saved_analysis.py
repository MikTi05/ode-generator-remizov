#!/usr/bin/env python3
"""Rebuild boundary tables/report from saved answers; no numerical solves."""
import argparse
import builtins
import json
from pathlib import Path
import shutil
import sys

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
parser.add_argument('--source',type=Path,default=Path(__file__).resolve().parent)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args();out=args.out.resolve();source=args.source.resolve()
if out.exists():raise FileExistsError('Use a NEW output directory')
sys.path.insert(0,str(args.root.resolve()))
original=builtins.__import__
def guard(name,*pos,**kw):
    blocked=('src','all_methods_dataset_loader','chernov_solver','remizov_solver','mc_solver','mc_optimized','ml','catboost','sklearn','torch')
    if any(name==prefix or name.startswith(prefix+'.') for prefix in blocked):raise RuntimeError('Forbidden import: '+name)
    return original(name,*pos,**kw)
builtins.__import__=guard
from diagnostics.report_boundary_profile_dev import report
from diagnostics.verify_boundary_profile_dev import verify
out.mkdir(parents=True)
for name in ('plan.json','started.json','session.json','selection.csv','call_journal.jsonl','editorial_note.md'):
    shutil.copyfile(source/name,out/name)
shutil.copytree(source/'records',out/'records')
report(out)
(out/'REPORT.md').write_text((out/'REPORT.md').read_text()+(out/'editorial_note.md').read_text())
verify(out)
