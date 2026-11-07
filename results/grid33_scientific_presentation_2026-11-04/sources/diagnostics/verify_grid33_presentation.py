#!/usr/bin/env python3
"""Independent scalar CSV-to-presentation verification; no producer/solver/ML import."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('results/saved_grid33_1000_2026-11-04')


def require(value,message):
    if not value: raise ValueError(message)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def table(path):
    with path.open(newline='',encoding='utf-8') as stream:return list(csv.DictReader(stream))


def check_tex(text):
    lines=[]
    for line in text.splitlines():
        cuts=[i for i,c in enumerate(line) if c=='%' and (len(line[:i])-len(line[:i].rstrip('\\')))%2==0]
        lines.append(line[:cuts[0]] if cuts else line)
    text='\n'.join(lines)
    active=lambda i: (len(text[:i])-len(text[:i].rstrip('\\'))) % 2 == 0
    balance=0
    for i,ch in enumerate(text):
        if ch in '{}' and active(i):
            balance+=1 if ch=='{' else -1
            require(balance>=0,'LaTeX negative brace depth')
    require(balance==0,'LaTeX braces')
    stack=[]
    for kind,name in re.findall(r'\\(begin|end)\{([^}]+)\}',text):
        if kind=='begin':stack.append(name)
        else:require(bool(stack) and stack.pop()==name,'LaTeX environments')
    require(not stack,'LaTeX open environments')
    labels=re.findall(r'\\label\{([^}]+)\}',text)
    require(len(labels)==len(set(labels)),'duplicate LaTeX labels')
    require(sum(c=='$' and active(i) for i,c in enumerate(text))%2==0,'LaTeX math delimiters')
    return labels


def verify(root,out):
    root,out=Path(root).resolve(),Path(out).resolve()
    inputs=json.loads((out/'input_sha256.json').read_text())
    require(all(sha(root/name)==digest for name,digest in inputs.items()),'source checksum')
    original=table(root/SOURCE/'grid33_all1000.csv')
    require(len(original)==len({r['equation_id'] for r in original})==1000,'source IDs')
    common=[r for r in original if r['R_E33']!='']
    def vals(rows,m):
        field='MC_E0_lower_bound' if m=='MC' else m+'_E33'
        return sorted(float(r[field]) for r in rows if r[field]!='')
    comparison=table(out/'comparison.csv')
    require(len(comparison)==8,'comparison rows')
    for scope,subset in (('all_available',original),('common997',common)):
        for m in ('R','RK4','PI','MC'):
            candidates=[r for r in comparison if r['scope']==scope and r['method']==m]
            require(len(candidates)==1,'unique scope/method')
            r=candidates[0];a=vals(subset,m);passed=sum(v<=.05 for v in a);failed=len(a)-passed
            p=.95*(len(a)-1);lo,hi=math.floor(p),math.ceil(p)
            expected=dict(n_values=len(a),n_scope=len(subset),missing_E33=len(subset) if m=='MC' else len(subset)-len(a),
                mean=statistics.mean(a),median=statistics.median(a),q95=a[lo]+(a[hi]-a[lo])*(p-lo),maximum=max(a),
                E33_pass=None if m=='MC' else passed,E33_fail=None if m=='MC' else failed,
                pass_fraction_available=None if m=='MC' else passed/len(a),
                pass_fraction_scope=None if m=='MC' else passed/len(subset),
                E33_fail_lower_bound=failed if m=='MC' else None,E33_unknown=passed if m=='MC' else len(subset)-len(a))
            require(r['metric']==('E0_lower_bound' if m=='MC' else 'E33'),'metric label')
            for key,value in expected.items():
                require(r[key]=='' if value is None else float(r[key])==value,'comparison cell '+m+'/'+key)
    data=json.loads((out/'figure_data.json').read_text())
    require(data['total']==1000 and data['common']==997,'figure counts')
    require(data['common_ids']==[r['equation_id'] for r in common],'common figure IDs')
    for m in ('R','RK4','PI','MC'):
        require(data['distributions_available'][m]==vals(original,m),'available figure values '+m)
        if m!='MC':require(data['distributions_common'][m]==vals(common,m),'common figure values '+m)
    missing=[r for r in original if r['R_E33']=='']
    require(table(out/'missing_remizov.csv')==missing,'missing rows preserved exactly')
    require(data['missing_ids']==[r['equation_id'] for r in missing],'missing ID list')
    expected_outliers=sorted([r for r in common if float(r['R_E33'])>.05],key=lambda r:(-float(r['R_E33']),r['equation_id']))
    outliers=table(out/'remizov_exceedances.csv')
    require(len(outliers)==len(data['exceedances'])==len(expected_outliers)==27,'R exceedances count')
    for r,published,plotted in zip(expected_outliers,outliers,data['exceedances']):
        for key in ('equation_id','family_id','R_E33','RK4_E33','PI_E33'):
            if key.endswith('E33'):
                require(float(r[key])==float(published[key])==plotted[key],'R exceedance value')
            else:require(r[key]==published[key]==plotted[key],'R exceedance identity')
    rv=vals(original,'R')
    share=(rv[-1]+rv[-2])/sum(rv)
    require(math.isclose(data['R_largest_two_sum_share'],share,rel_tol=1e-14),'R sum share')
    require(data['R_above_one']==sum(v>1 for v in rv)==4,'R above1')
    shares=table(out/'threshold_shares.csv')
    require(len(shares)==4,'shares rows')
    for row in shares:
        m=row['method'];a=vals(original,m);mc=m=='MC'
        expected=dict(denominator=1000,measured_pass=None if mc else sum(v<=.05 for v in a),
            measured_fail=None if mc else sum(v>.05 for v in a),missing_answer=0 if mc else 1000-len(a),
            fails_by_lower_bound=sum(v>.05 for v in a) if mc else 0,
            unknown_grid_accuracy=sum(v<=.05 for v in a) if mc else 0,
            full_grid_error_missing=1000 if mc else 1000-len(a))
        plotted=next(r for r in data['shares'] if r['method']==m)
        for key,value in expected.items():
            require(row[key]=='' if value is None else int(row[key])==value,'share cell '+m+'/'+key)
            require(plotted[key]==value,'plotted share '+m+'/'+key)
        require(sum(value or 0 for key,value in expected.items() if key not in ('denominator','full_grid_error_missing'))==1000,'share denominator')
    for m in ('R','PI','MC'):
        key='MC_E0_lower_bound' if m=='MC' else m+'_E33'
        margin,pid=min((float(r[key])-float(r['RK4_E33']),r['equation_id']) for r in common)
        require(margin>0 and data['minimum_margins'][m]==dict(margin=margin,equation_id=pid),'strict margin')
    section=(out/'section.md').read_text();tex=(out/'section.tex').read_text()
    for token in ('970/997','1000/1000','404/1000','402/997','97.2919%','98.8426%',
                  '62','938','27','39,4119','0,00274797'):
        require(token in section,'missing or inconsistent reported fact: '+token)
    for r in comparison:
        for key in ('mean','median','q95','maximum'):
            require(f"{float(r[key]):.6g}" in section,'table rounding inconsistent with CSV')
    markdown_rows=[line.strip('|').split('|') for line in section.splitlines()
                   if line.startswith('| ') and not line.startswith('| Метод')]
    require(len(markdown_rows)==8,'Markdown comparison rows')
    def texnum(value):
        token=f'{value:.6g}'
        if 'e' in token:
            a,b=token.split('e');return a+r'\times10^{'+str(int(b))+'}'
        return token
    for scope,filename in (('all_available','comparison_table.tex'),('common997','comparison_common997.tex')):
        latex_rows=[line.split('&') for line in (out/filename).read_text().splitlines()
                    if '&' in line and not line.startswith('Метод')]
        require(len(latex_rows)==4,'LaTeX comparison rows')
        for published,r in zip(latex_rows,[r for r in comparison if r['scope']==scope]):
            cells=[c.strip() for c in published]
            md=[c.strip() for c in markdown_rows[comparison.index(r)]]
            require(cells[2]==md[2]==r['n_values'],'report table n')
            for offset,key in enumerate(('mean','median','q95','maximum'),3):
                require(cells[offset]=='$'+texnum(float(r[key]))+'$','LaTeX rounded statistic')
                require(md[offset]==f'{float(r[key]):.6g}','Markdown rounded statistic')
            if r['method']=='MC':
                require('E_0' in cells[1] and 'нижняя граница' in md[1],'MC table type')
                require(f">= {r['E33_fail_lower_bound']}".replace('>=',r'\ge') in cells[-1]
                        and f"{r['E33_unknown']} неизвестных" in cells[-1],'LaTeX MC unknown/fail')
                require(md[-1]==f"≥{r['E33_fail_lower_bound']} нарушений; {r['E33_unknown']} неизвестных",'Markdown MC unknown/fail')
            else:
                fraction=f"{r['E33_pass']}/{r['n_values']}"
                require(fraction in cells[-1] and md[-1]==fraction+f" ({100*float(r['pass_fraction_available']):.4f}%)",'report fractions')
    require('нижняя граница' in section and 'новую случайную реализацию' in section and
            'скорость сходимости' in section and 'Универсальное превосходство RK4 не утверждается' in section,
            'scientific qualifications absent')
    labels=check_tex(tex+'\n'+(out/'comparison_table.tex').read_text()+'\n'+(out/'comparison_common997.tex').read_text())
    old=(root/'results/scientific_report_revision_2026-11-04/report.tex').read_text()
    old_labels=set(re.findall(r'\\label\{([^}]+)\}',old))
    require(not old_labels.intersection(labels),'LaTeX label collides with old report')
    for path in re.findall(r'\\(?:input|includegraphics)(?:\[[^\]]*\])?\{([^}]+)\}',tex):
        require(path.startswith('grid33_supplement/') and (out/path.removeprefix('grid33_supplement/')).is_file(),'LaTeX input path')
    requirements=table(out/'requirements.csv')
    require(len(requirements)==len({r['id'] for r in requirements})==14,'requirements traceability')
    require('0,01' in next(r['requirement'] for r in requirements if r['id']=='T08'),'historical tolerance lost')
    require('не закрыто' in next(r['status'] for r in requirements if r['id']=='T08'),'historical speed task falsely closed')
    manifest=json.loads((out/'figure_manifest.json').read_text());require(len(manifest['figures'])==4,'figure count')
    from PIL import Image
    for f in manifest['figures']:
        for item in f['files']:
            path=out/item['path'];require(sha(path)==item['sha256'],'figure checksum')
            if path.suffix=='.png':
                with Image.open(path) as im:require(im.width>=1400 and im.height>=900,'figure resolution')
            else:require(path.read_bytes().startswith(b'%PDF-'),'PDF signature')
    require(all(sha(root/name)==digest for name,digest in inputs.items()),'source changed during verification')
    names=('comparison.csv','remizov_exceedances.csv','threshold_shares.csv','missing_remizov.csv',
           'figure_data.json','figure_manifest.json','requirements.csv','section.md','section.tex',
           'comparison_table.tex','comparison_common997.tex','comparison.md','REQUIREMENTS.md',
           'README.md','input_sha256.json','environment.json')
    artifacts=[out/name for name in names]+[out/item['path'] for f in manifest['figures'] for item in f['files']]
    result=dict(passed=True,source_rows=1000,comparison_rows_checked=8,
        comparison_statistic_cells_checked=8*13,distribution_values_checked=3*997+997+3*1000,
        R_exceedances_checked=27,missing_R_rows_checked=3,share_rows_checked=4,
        figures_checked=4,figure_files_checked=8,latex_static_checked=True,latex_unique_labels=len(labels),
        latex_compiled=False,formatted_MD_and_TeX_tables_checked=True,requirements_checked=14,new_solver_calls=0,new_MC_calls=0,new_fit_calls=0,new_model_inference_calls=0,
        artifacts_sha256={str(p.relative_to(out)):sha(p) for p in sorted(artifacts)})
    (out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='artifacts_sha256'},ensure_ascii=False))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();verify(args.root,args.out)
