#!/usr/bin/env python3
"""Reproduce final tables and report from saved attempts; no solver or ML imports."""
from pathlib import Path
import argparse
import builtins
import json
import shutil
import sys

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
parser.add_argument('--source',type=Path,default=Path(__file__).resolve().parent)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args();source=args.source.resolve();out=args.out.resolve()
if out.exists():raise FileExistsError('Use a new --out directory')
sys.path.insert(0,str(args.root.resolve()))
original_import=builtins.__import__
def guard(name,*pos,**kw):
    blocked=('src','all_methods_dataset_loader','remizov_solver','chernov_solver','mc_solver','mc_optimized','ml','catboost','sklearn','torch')
    if any(name==p or name.startswith(p+'.') for p in blocked):raise RuntimeError('Forbidden import: '+name)
    return original_import(name,*pos,**kw)
builtins.__import__=guard
from diagnostics.report_cost60 import build
from diagnostics.verify_cost60 import verify
out.mkdir(parents=True)
for name in ('plan.json','session.json','selection.csv','candidate_pool.csv','started.json','environment.before.json','editorial_patch.json'):
    shutil.copyfile(source/name,out/name)
for folder in ('records','inputs'):
    shutil.copytree(source/folder,out/folder)
build(out)
patch=json.loads((out/'editorial_patch.json').read_text())
text=(out/'REPORT.md').read_text()
for old,new in patch['replace']:
    if old not in text:raise ValueError('Frozen editorial patch no longer matches generated report')
    text=text.replace(old,new)
(out/'REPORT.md').write_text(text+patch['append'])
verify(out)
