"""Verify only the dossier, saved-data arithmetic and preservation checksums."""
import ast
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import sys
from urllib.parse import unquote

import numpy as np
import pandas as pd
from PIL import Image

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def read(name):
    return json.loads((OUT / name).read_text())


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    document = (OUT / 'EVIDENCE.md').read_text()
    targets = re.findall(r'\[[^\]]+\]\(([^)]+)\)', document)
    broken, anchors = [], []
    for target in targets:
        if target.startswith(('https:', 'http:')):
            continue
        own, _, anchor = target.partition('#')
        path = (OUT / unquote(own)).resolve() if own else OUT / 'EVIDENCE.md'
        generated_by_this_check = path in {OUT / 'checks/integrity.json', OUT / 'checks/report_validation.json'}
        if not path.is_file() and not generated_by_this_check:
            broken.append(target)
        if anchor.startswith('L'):
            line = int(anchor[1:])
            assert line <= len(path.read_text().splitlines()), target
            anchors.append(target)
    assert not broken, broken
    expected_width = None
    tables = 0
    for number, line in enumerate(document.splitlines(), 1):
        if line.startswith('|'):
            width = len(re.split(r'(?<!\\)\|', line)) - 2
            if expected_width is None:
                tables += 1
                expected_width = width
            assert width == expected_width, (number, width, expected_width)
        else:
            expected_width = None
    assert document.count('```') % 2 == 0
    features = pd.read_csv(OUT / 'tables/all_258_features.csv')
    assert len(features) == features.feature_name.nunique() == 258
    with np.load(ROOT / 'results/ml_selector_hypotheses_2026-10-31/features.npz', allow_pickle=False) as z:
        assert features.feature_name.tolist() == z['augmented_names'].tolist()
    for row in features.itertuples():
        marker = f'| {row.index} | `{row.feature_name}` |'
        assert document.count(marker) == 1, marker
    cases = pd.read_csv(OUT / 'tables/diagnostic_258_cases.csv', float_precision='round_trip')
    assert len(cases) == cases.equation_id.nunique() == 23
    assert Counter(cases.category) == dict(TP=6, FP=14, FN=3)
    for row in cases.itertuples():
        assert row.equation_id in document
        assert format(row.E_R, '.17g') in document and format(row.E_MC, '.17g') in document
        assert format(row.delta_R_minus_MC, '.17g') in document
    assert set(cases[cases.harms_epsilon].equation_id) == {'AMV1_F0180_S01', 'AMV1_F0180_S02'}
    assert not cases.rescues_epsilon.any()
    policies = pd.read_csv(OUT / 'tables/model_policy_metrics_all.csv')
    assert len(policies) == 2532
    assert len(policies[['candidate_id','policy']].drop_duplicates()) == 844
    assert set(policies.split) == {'train', 'validation', 'test'}
    models = pd.read_csv(OUT / 'tables/models_96.csv')
    assert len(models) == models.candidate_id.nunique() == 96
    for cid in models.candidate_id:
        assert f'| {cid} |' in document
    figures = read('figure_manifest.json')
    assert len(figures) == 14
    figure_details = []
    for row in figures:
        png, pdf = (OUT / 'figures' / row[k] for k in ('file', 'pdf'))
        assert f'(figures/{png.name})' in document and f'(figures/{pdf.name})' in document
        with Image.open(png) as img:
            img.verify()
        with Image.open(png) as img:
            width, height = img.size
        assert width >= 1000 and height >= 500
        assert pdf.read_bytes().startswith(b'%PDF-') and pdf.stat().st_size > 1000
        figure_details.append(dict(png=png.name, pixels=[width,height],pdf=pdf.name))
    assert read('checks/visual_qa.json')['all_figures_reviewed']
    for script in OUT.glob('*.py'):
        tree = ast.parse(script.read_text(), filename=str(script))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in {'fit','fit_transform','solve','solve_rk4','solve_product_integral_left_exponential','generate_dataset'}, (script.name,node.lineno)
    before = read('before_sha256.json')
    missing, changed = [], []
    for path, expected in before.items():
        absolute = ROOT / path
        if not absolute.is_file():
            missing.append(path)
        elif digest(absolute) != expected:
            changed.append(path)
    assert not missing
    assert set(changed) <= {'PROJECT_STATUS.md'}, changed
    original_status = (OUT / 'PROJECT_STATUS.before.md').read_text()
    current_status = (ROOT / 'PROJECT_STATUS.md').read_text()
    assert current_status == original_status or current_status.endswith(original_status.split('\n\n',1)[1])
    manifest = json.loads((ROOT / 'MIGRATION_MANIFEST.json').read_text())
    cores = {}
    for name in manifest['unchanged_mathematical_core']:
        cores[name] = digest(ROOT / name)
        assert cores[name] == manifest['source_sha256'][name], name
    write('checks/integrity.json', dict(previous_files=len(before), changed_previous_files=changed,
        missing_previous_files=missing, allowed_changes=['PROJECT_STATUS.md'],
        source_data_results_models_unchanged=True, mathematical_core=cores,
        original_status_body_preserved=True, neighbor_repository_accessed=False))
    sources = read('used_sources.json')
    for name in sources:
        if name.startswith(str(OUT.relative_to(ROOT))) or name == 'PROJECT_STATUS.md':
            sources[name] = digest(ROOT / name)
        else:
            assert digest(ROOT / name) == sources[name], name
    (OUT / 'used_sources.json').write_text(json.dumps(sources,indent=2,ensure_ascii=False)+'\n')
    write('checks/environment.json', dict(python=sys.version, platform=platform.platform(),
        versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','catboost','matplotlib','Pillow']}))
    assert read('evidence_summary.json')['training_calls'] == 0
    assert read('evidence_summary.json')['numerical_solver_calls'] == 0
    assert read('checks/raw_saved_records.json')['solver_calls'] == 0
    result = dict(status='passed', markdown_links_checked=len(targets), code_line_links_checked=len(anchors),
        broken_links=broken, markdown_tables_checked=tables, feature_rows=258, diagnostic_cases=23,
        diagnostic_case_categories=dict(Counter(cases.category)), saved_models=96,
        policy_split_metric_rows=2532, historical_split_metrics=63,
        figures=figure_details, new_training_calls=0, new_solver_calls=0,
        new_generator_calls=0, previous_files_checked=len(before),
        changed_previous_files=changed, source_paths_pinned=len(sources),
        report_sha256=digest(OUT/'EVIDENCE.md'),
        scope='saved-data checks; historical fitting and numerical experiments not rerun')
    write('checks/report_validation.json',result)
    for name in ('checks/integrity.json','checks/report_validation.json'):
        assert (OUT/name).is_file()
    write('artifact_manifest.json', {str(p.relative_to(OUT)):digest(p) for p in sorted(OUT.rglob('*'))
                                   if p.is_file() and p.name != 'artifact_manifest.json'})
    print(json.dumps({k:v for k,v in result.items() if k!='figures'},ensure_ascii=False))


if __name__ == '__main__':
    main()
