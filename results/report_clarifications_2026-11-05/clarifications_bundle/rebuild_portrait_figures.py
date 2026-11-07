"""Four portrait figures using frozen figure_data.json; no equation/model code."""
from pathlib import Path
import hashlib
import json
import math
import os
import tempfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=HERE/'plot_data.json'
EPS=.05
WIDTH=16.5/2.54
COLORS=dict(R='#8b479d',RK4='#226bb0',PI='#c77522',MC='#47786a')
LABELS=dict(R='Метод Ремизова',RK4='RK4',PI='Метод мультипликативного\nинтеграла',MC='Монте-Карло')

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def build():
    data=json.loads(SOURCE.read_text())
    assert len(data['common_ids'])==997 and len(data['exceedances'])==27
    with tempfile.TemporaryDirectory(prefix='2488-portrait-mpl-') as cache:
        os.environ['MPLCONFIGDIR']=cache
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import numpy as np
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
            'axes.titlesize':11,'axes.labelsize':10,'xtick.labelsize':10,'ytick.labelsize':10,
            'legend.fontsize':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
        manifest=[]
        def save(fig,name,observations):
            # Do not use bbox_inches='tight': the physical width must remain 16.5 cm.
            fig.canvas.draw()
            renderer=fig.canvas.get_renderer();width,height=fig.canvas.get_width_height()
            overflow=[]
            for obj in fig.findobj(matplotlib.text.Text):
                if obj.get_visible() and obj.get_text():
                    assert obj.get_fontsize()>=10,obj.get_text()
                    b=obj.get_window_extent(renderer)
                    if b.x0 < -.5 or b.y0 < -.5 or b.x1 > width+.5 or b.y1 > height+.5:
                        overflow.append(obj.get_text())
            assert not overflow,overflow
            files=[]
            for ext in ('pdf','png'):
                path=HERE/'figures'/(name+'.'+ext)
                fig.savefig(path,dpi=180,facecolor='white',
                    metadata={'Creator':'NIR2488 saved-data portrait figures',**({'CreationDate':None,'ModDate':None} if ext=='pdf' else {})})
                files.append(dict(path=str(path.relative_to(HERE)),sha256=sha(path)))
            manifest.append(dict(name=name,width_cm=fig.get_figwidth()*2.54,height_cm=fig.get_figheight()*2.54,
                                 minimum_text_size_pt=10,observations=observations,files=files))
            plt.close(fig)
        def logaxis(ax,ticks):
            ax.set_xscale('log');ax.set_xticks([10.**k for k in ticks])
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v,_:('1' if v==1 else '1e'+str(round(math.log10(v))))))
            ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        all_values=[v for a in data['distributions_available'].values() for v in a]
        assert min(all_values)>0
        lo,hi=10**math.floor(math.log10(min(all_values))),10**math.ceil(math.log10(max(all_values)))
        fig,ax=plt.subplots(figsize=(WIDTH,6.1));fig.subplots_adjust(left=.14,right=.97,bottom=.34,top=.88)
        for m,a in data['distributions_common'].items():
            a=np.asarray(a);y=np.arange(1,len(a)+1)/len(a)
            ax.step(np.r_[lo,a,hi],np.r_[0,y,1],where='post',label=LABELS[m].replace('\n',' '),color=COLORS[m],lw=1.7)
        ax.axvline(EPS,color='#a1272d',ls='--',label='Точность 0.05')
        logaxis(ax,range(-10,5,2));ax.set_xlim(lo,hi);ax.set_ylim(0,1.02);ax.set_yticks(np.linspace(0,1,6))
        ax.set_xlabel('Наибольшая абсолютная ошибка E₃₃\n(логарифмическая шкала)')
        ax.set_ylabel('Доля уравнений с ошибкой ≤ E₃₃');ax.set_title('Распределение E₃₃\n997 общих уравнений',pad=12)
        ax.grid(alpha=.22)
        handles,labels=ax.get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.55,.015),frameon=False)
        save(fig,'grid33_01_ecdf_common997',{m:len(a) for m,a in data['distributions_common'].items()})

        fig,axes=plt.subplots(2,2,figsize=(WIDTH,7.35),sharex=True,sharey=True)
        fig.subplots_adjust(left=.13,right=.97,bottom=.14,top=.85,wspace=.17,hspace=.50)
        bins=np.geomspace(lo,hi,37)
        for ax,m in zip(axes.flat,('R','RK4','PI','MC')):
            a=np.asarray(data['distributions_available'][m]);counts,edges=np.histogram(a,bins=bins)
            assert int(counts.sum())==len(a)
            ax.stairs(counts/len(a),edges,fill=True,color=COLORS[m],alpha=.75)
            ax.axvline(EPS,color='#a1272d',ls='--');logaxis(ax,(-9,-6,-3,0,3));ax.set_xlim(lo,hi)
            ax.set_title(LABELS[m]+f'\nn = {len(a)}'+('\nE₀: нижняя оценка E₃₃' if m=='MC' else '\nошибка E₃₃'),fontsize=10,pad=8)
            ax.set_yticks([0,.1,.2,.3,.4]);ax.grid(alpha=.18)
        fig.suptitle('Распределения по доступным ответам\nМонте-Карло показан отдельно',fontsize=12,y=.98)
        fig.supxlabel('Ошибка (логарифмическая шкала)\nОдинаковые интервалы; красная линия: 0.05',fontsize=10,y=.025)
        fig.supylabel('Доля наблюдений в интервале (не плотность)',fontsize=10,x=.02)
        save(fig,'grid33_02_histograms_available',{m:len(a) for m,a in data['distributions_available'].items()})

        e=data['exceedances'];fig,ax=plt.subplots(figsize=(WIDTH,7.65))
        fig.subplots_adjust(left=.265,right=.96,bottom=.11,top=.89)
        yy=np.arange(len(e));xx=[r['R_E33'] for r in e]
        ax.hlines(yy,EPS,xx,color='#c9b8ce',lw=.8)
        ax.scatter(xx,yy,c=['#b62e3a' if i<2 else COLORS['R'] for i in range(len(e))],s=22,zorder=3)
        for i,v in enumerate(xx):ax.annotate(f'{v:.3g}',(v,i),xytext=(4,0),textcoords='offset points',va='center',fontsize=10)
        ax.set_yticks(yy,[r['equation_id'] for r in e],fontsize=10);ax.invert_yaxis()
        logaxis(ax,range(-1,5));ax.set_xlim(EPS*.8,max(xx)*8)
        ax.set_xlabel('Ошибка E₃₃ (логарифмическая шкала)')
        ax.set_title('Метод Ремизова: 27 ошибок E₃₃ > 0.05\n997 сохранённых ответов',fontsize=11,pad=12)
        ax.axvline(EPS,color='#a1272d',ls='--');ax.grid(axis='x',alpha=.22)
        save(fig,'grid33_03_remizov_exceedances',{'R_exceedances':len(e)})

        fig,ax=plt.subplots(figsize=(WIDTH,6.6));fig.subplots_adjust(left=.27,right=.97,bottom=.35,top=.88)
        y=np.arange(4);left=np.zeros(4)
        cats=[('measured_pass','Измерено E₃₃ ≤ 0.05','#397963',''),('measured_fail','Измерено E₃₃ > 0.05','#bd5761',''),
              ('missing_answer','Ответ отсутствует','#777b85','//'),('fails_by_lower_bound','E₀ > 0.05: точность не достигнута','#e2a260','..'),
              ('unknown_grid_accuracy','Точность по 33 узлам не установлена','#dfe3e9','xx')]
        for key,label,color,hatch in cats:
            counts=np.array([r[key] or 0 for r in data['shares']]);width=counts/10
            ax.barh(y,width,left=left,label=label,color=color,hatch=hatch,edgecolor='white',height=.57)
            for i,(count,w,l) in enumerate(zip(counts,width,left)):
                if count>=80:
                    ax.text(l+w/2,i,f'{count}/1000\n{w:g}%',ha='center',va='center',fontsize=10,color='#222' if key=='unknown_grid_accuracy' else 'white')
            left+=width
        assert np.all(left==100)
        ax.set_yticks(y,['Метод Ремизова','RK4','Метод\nмультипликативного\nинтеграла','Монте-Карло\n(только E₀)'],fontsize=10)
        ax.set_ylim(3.8,-.9);ax.set_xlim(0,100);ax.set_xticks([0,20,40,60,80,100])
        ax.text(50,-.62,'27 ошибок > 0.05; 3 ответа отсутствуют',ha='center',fontsize=10)
        ax.annotate('62/1000 (6.2%)',xy=(3.1,3.20),xytext=(18,3.65),fontsize=10,
                    arrowprops=dict(arrowstyle='-',color='#555',lw=.7),ha='center')
        ax.set_xlabel('Доля всех 1000 уравнений, %');ax.set_title('Достижение точности 0.05\nОбщее число уравнений: 1000',fontsize=12,pad=12)
        handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.53,.02),ncol=1,frameon=False,fontsize=10)
        save(fig,'grid33_04_threshold_shares',{'denominator':1000,'R_pass':970,'RK4_pass':1000,'PI_pass':404,'MC_bound_fail':62,'MC_unknown':938})
    (HERE/'figure_manifest.json').write_text(json.dumps({'source':SOURCE.name,'source_origin':'results/grid33_scientific_presentation_2026-11-04/figure_data.json','source_sha256':sha(SOURCE),
        'same_saved_data':True,'histogram_bins_unchanged':36,'no_shift_trim_or_recalculation_of_errors':True,
        'figures':manifest,'new_solver_MC_ML_calls':0},ensure_ascii=False,indent=2)+'\n')
    print('4 portrait PDF + 4 PNG; saved data only')

if __name__=='__main__':build()
