"""Publication plots of saved error metrics; no ODE calculations."""
import argparse
import csv
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/mpl_remizov_report_20261020')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report-dir',required=True,type=Path)
    root=p.parse_args().report_dir
    rows=list(csv.DictReader((root/'analysis/all_1000.csv').open()))
    errors=np.sort([float(r['error']) for r in rows if r['error']])
    groups=json.loads((root/'analysis/analysis.json').read_text())['by_cell']
    regimes=['constant','slow_varying','moderate_varying','oscillatory','high_contrast']
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    ax=axes[0];ax.semilogy(np.arange(1,len(errors)+1),errors,color='#185b75',lw=1.8)
    ax.axhline(.01,color='#a53e30',ls='--',lw=1.3,label='Порог 0.01')
    ax.scatter(np.arange(995,998),errors[-3:],color='#a53e30',s=26,zorder=3,label='Три экстремальных выброса')
    ax.set(xlabel='Ранг ошибки среди 997 ответов',ylabel='E∞ на 33 точках',title='Распределение фактической ошибки')
    ax.grid(alpha=.2,which='both');ax.legend(fontsize=8,loc='upper left')
    values=np.array([[groups[f'L{i}/{mode}']['outcomes'].get('accuracy_not_met',0)/50 for mode in regimes] for i in range(1,5)])
    ax=axes[1];im=ax.imshow(values,vmin=0,vmax=.7,cmap='YlOrRd',aspect='auto')
    for i in range(4):
        for j in range(5):
            count=groups[f'L{i+1}/{regimes[j]}']['outcomes'].get('accuracy_not_met',0)
            text=f'{count}/50\n{100*values[i,j]:.0f}%'+('\n+3 ошибки' if (i,j)==(2,3) else '')
            ax.text(j,i,text,ha='center',va='center',fontsize=8,color='white' if values[i,j]>.4 else '#151515')
    ax.set_xticks(range(5),['constant','slow','moderate','oscillatory','contrast'],rotation=28,ha='right')
    ax.set_yticks(range(4),['L1','L2','L3','L4']);ax.set_title('Ответы выше порога: число и доля\nв каждой ячейке 50 уравнений')
    fig.colorbar(im,ax=ax,label='Доля accuracy_not_met',shrink=.8)
    for suffix in ('png','svg'):
        target=root/f'errors_overview.{suffix}'
        if target.exists():raise FileExistsError(target)
        fig.savefig(target,dpi=180)
    plt.close(fig)
    print('Saved errors_overview.png and errors_overview.svg')


if __name__=='__main__':main()
