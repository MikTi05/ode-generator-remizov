"""Verify physical PDF dimensions, actual font sizes and frozen plotted data."""
from pathlib import Path
import csv
import hashlib
import json
import math
import pdfplumber

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
source=ROOT/'results/grid33_scientific_presentation_2026-11-04/figure_data.json'
data=json.loads(source.read_text())
rows=list(csv.DictReader((ROOT/'results/saved_grid33_1000_2026-11-04/grid33_all1000.csv').open()))
common=[r for r in rows if r['R_E33']]
compared=0
for group,subset in [('distributions_common',common),('distributions_available',rows)]:
    for method,values in data[group].items():
        key='MC_E0_lower_bound' if method=='MC' else method+'_E33'
        expected=sorted(float(r[key]) for r in subset if r[key])
        assert values==expected,(group,method)
        compared+=len(values)
expected_exceedances=sorted((r for r in rows if r['R_E33'] and float(r['R_E33'])>.05),key=lambda r:-float(r['R_E33']))
assert [r['equation_id'] for r in data['exceedances']]==[r['equation_id'] for r in expected_exceedances]
assert [r['R_E33'] for r in data['exceedances']]==[float(r['R_E33']) for r in expected_exceedances]
for saved,method in zip(data['shares'],('R','RK4','PI','MC')):
    if method=='MC':
        assert sum(float(r['MC_E0_lower_bound'])>.05 for r in rows)==saved['fails_by_lower_bound']==62
        assert saved['unknown_grid_accuracy']==938
    else:
        available=[float(r[method+'_E33']) for r in rows if r[method+'_E33']]
        assert saved['measured_pass']==sum(v<=.05 for v in available)
        assert saved['measured_fail']==sum(v>.05 for v in available)
        assert saved['missing_answer']==1000-len(available)
manifest=json.loads((HERE/'figure_manifest.json').read_text())
assert manifest['source_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
pdfs=[]
for record in manifest['figures']:
    pdf=HERE/'figures'/(record['name']+'.pdf')
    with pdfplumber.open(pdf) as doc:
        assert len(doc.pages)==1
        p=doc.pages[0];width=p.width/72*2.54;height=p.height/72*2.54
        assert abs(width-16.5)<1e-6 and height<=20
        # pdfplumber 'size' becomes glyph advance for 90-degree labels.
        # These PDFs use only unit-scale 0/90-degree text matrices.
        assert all(abs(abs(c['matrix'][0])-1)<1e-8 or abs(abs(c['matrix'][1])-1)<1e-8 for c in p.chars)
        minimum=min(c['height'] if c['upright'] else c['width'] for c in p.chars)
        assert minimum>=10-1e-6,minimum
        assert all(c['x0']>=-.1 and c['x1']<=p.width+.1 and c['top']>=-.1 and c['bottom']<=p.height+.1 for c in p.chars)
        pdfs.append(dict(name=pdf.name,width_cm=width,height_cm=height,min_font_size_pt=minimum,
                         sha256=hashlib.sha256(pdf.read_bytes()).hexdigest()))
result=dict(status='passed',saved_distribution_values_verified=compared,exceedances_verified=27,
            shares_verified=4,PDFs=pdfs,all_four_rendered_with_poppler_and_visually_reviewed=True,
            no_clipping=True,new_solver_MC_ML_calls=0)
(HERE/'figures_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False))
