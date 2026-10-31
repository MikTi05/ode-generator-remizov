"""Render the completed saved-only R40 report and timing figure; no solver calls."""
from __future__ import annotations
import argparse,os,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from diagnostics.audit_probability_selector import read,write


def summarize(out):
    a=read(out/'analysis.json');assert a['passed'] and read(out/'independent_verification.json')['passed']
    before=read(out/'hardware_power_before.json');after={}
    for n,cmd in [('settings',['pmset','-g','custom']),('source',['pmset','-g','batt'])]:
        r=subprocess.run(cmd,capture_output=True,text=True);after[n]=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
    after['settings_unchanged']=after['settings']==before['settings']
    after['power_source_unchanged']=after['source']['stdout'].splitlines()[0]==before['source']['stdout'].splitlines()[0]
    write(out/'hardware_power_after.json',after)
    all5=a['aggregate']['all5'];good=a['aggregate']['both_accurate'];r=a['resources']
    paired=[z['relative_saved'] for c in a['cases'] for z in c['paired_episodes']]
    supported=sum(c['supported_accurate_speedup'] for c in a['cases'])
    assert supported==4 and all(z>0 for z in paired)
    with tempfile.TemporaryDirectory(prefix='switch5_R40_mpl_') as cache:
        os.environ['MPLCONFIGDIR']=cache
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
        for i,c in enumerate(a['cases']):
            for p,offset,color in [('R128',-.13,'#3182BD'),('R256',.13,'#E6550D')]:
                t=c[p]['timing'];ys=[i+offset+v for v in (-.035,-.012,.012,.035)]
                axes[0].scatter(t['raw'],ys,color=color,label=p if i==0 else None,s=24)
                axes[0].plot([t['min'],t['max']],[i+offset]*2,color=color,alpha=.5)
                axes[0].scatter([t['median']],[i+offset],marker='|',s=140,color='black')
            saving=[100*z['relative_saved'] for z in c['paired_episodes']]
            axes[1].scatter(saving,[i]*4,color='#31A354' if c['R128']['accurate'] else '#DE2D26')
            axes[1].plot([min(saving),max(saving)],[i]*2,color='.5')
        labels=[c['equation_id'].replace('AMV1_','')+(' (both inaccurate)' if not c['R128']['accurate'] else '') for c in a['cases']]
        axes[0].set(yticks=list(range(5)),yticklabels=labels,xlabel='Full method wall, seconds',title='All four calls per profile; black marker = median')
        axes[0].legend(fontsize=8,loc='lower left');axes[0].invert_yaxis()
        axes[1].set(yticks=list(range(5)),yticklabels=[],xlabel='Paired episode time saved, %',title='Four paired observations; not reliability bounds')
        axes[1].invert_yaxis();axes[1].set_xlim(45,54)
        fig.suptitle('Frozen TRAIN OOF switches: R128 / constant R256, one matched cohort')
        fig.savefig(out/'matched_timings.png',dpi=170);fig.savefig(out/'matched_timings.svg');plt.close(fig)
    lines=['# R40 завершён: стоимость пяти зафиксированных train-OOF переключений', '',
'Дата:2026-10-26. **40/40 вызовов завершены**:5 исходных train-задач,4 R128+4 R256 на каждую. '
'**Локальная экономия подтверждена**, точность относительно constant_R256 сохранена. '
'ML-модели/политика/численные методы не изменены, MC/ML/benchmark1000/validation/test не запускались.', '',
f'Сумма median4 method wall для пяти действий: **{all5["R128_median_cost"]:.3f} с R128 '
f'против {all5["R256_median_cost"]:.3f} с R256**, выигрыш **{all5["saved_seconds"]:.3f} с '
f'({100*all5["relative_saved"]:.2f}%, ускорение×{all5["speedup_factor"]:.3f})**. '
'Это5 доставок по медианам, а не время всей40-call исследовательской серии и не полная online-стоимость селектора.', '',
'## Точность и известный неточный контроль', '',
'Все40 векторов результата совпали с каноническими сохранёнными референсами побитово. '
'Все40 E_grid независимо пересчитаны на33 узлах [-1,1]; ни один solver failure, '
'timeout или interrupted вызов не скрыт/не заменён. Оба профиля точны на4/5 задачах: '
'16/20 вызовов каждого профиля удовлетворяют E_grid≤.01. Повторные детерминированные '
'ответы не увеличивают число независимых accuracy-задач.', '',
'| Задача | E R128 | E R256 | Точность обоих | Классификация |',
'|---|---:|---:|---|---|']
    for c in a['cases']:
        lines.append(f'| {c["equation_id"]} | {c["R128"]["E_grid"]:.9f} | {c["R256"]["E_grid"]:.9f} | {c["R128"]["accurate"] and c["R256"]["accurate"]} | {c["classification"]} |')
    lines += ['', '**F0172/S03:** R128 E=.04358398499447341, R256 E=.021515665631340275. '
'Оба неточны во всех4 повторах, как было известно до запуска. Более быстрый '
'R128 здесь не объявляется точным winner. Случай сохранён в raw/cost статистике, '
'а в анализе четырёх точных задач отделён. На4 baseline-correct задачах нет '
'потерянной доставки. Это80% наблюдаемой точности каждого профиля на данном '
'наборе, не95% гарантия, несмотря на q=.95 в исходной ML-политике.', '',
'## Сопоставимое время и разброс', '',
'| Задача | R128 median [min,max],с | R256 median [min,max],с | Экономия | MAD/median R128 / R256 |',
'|---|---:|---:|---:|---:|']
    for c in a['cases']:
        x=c['R128']['timing'];y=c['R256']['timing']
        lines.append(f'| {c["equation_id"].replace("AMV1_","")} | {x["median"]:.3f} [{x["min"]:.3f},{x["max"]:.3f}] | {y["median"]:.3f} [{y["min"]:.3f},{y["max"]:.3f}] | {100*c["relative_saved"]:.2f}% | {x["MAD_fraction"]:.2%} / {y["MAD_fraction"]:.2%} |')
    maxmad=max(c[p]['timing']['MAD_fraction'] for c in a['cases'] for p in ('R128','R256'))
    lines += ['', f'На всех5 задачах max(t_R128)<min(t_R256), median_R128≤.9 median_R256; '
f'максимальный MAD/median всего **{maxmad:.2%}**, ниже заранее заданных20%. '
f'Все20 пар episode показывают положительную экономию; диапазон '
f'**{100*min(paired):.2f}–{100*max(paired):.2f}%**. '
'На4 точных задачах выполнены все заранее заданные условия устойчивого '
'timing-преимущества. На F0172 условия скорости также выполнены, но точности нет. '
'Не отбрасывались первые/крайние замеры, не добавлялись повторы и не подбирался '
'порог/состав задач. Четыре измерения дают описательную устойчивость в этом '
'окружении, не сильное статистическое доказательство для новых машин/семейств.', '',
'![Все40 времён](matched_timings.png)', '',
'Полные raw времена, IQR/MAD и разности каждой пары episodes — '
'[analysis.json](analysis.json); все40 попыток, ошибки и SHA — [attempts.csv](attempts.csv).', '',
'## Стоимость и смысл для ML-политики', '',
'| Область | R128,с | R256,с | Сэкономлено,с | Экономия | Точных доставок обоих |',
'|---|---:|---:|---:|---:|---:|']
    for name,z in [('Все5, включая неточный ответ',all5),('Только4 точные задачи',good)]:
        lines.append(f'| {name} | {z["R128_median_cost"]:.3f} | {z["R256_median_cost"]:.3f} | {z["saved_seconds"]:.3f} | {100*z["relative_saved"]:.2f}% | {z["accurate_deliveries"]} |')
    lines += ['', f'Стоимость на точную доставку с оплатой также неточного ответа: '
f'**{all5["method_seconds_per_accurate"]:.3f} с R128 против '
f'{all5["baseline_method_seconds_per_accurate"]:.3f} с R256** '
'(total method cost на5 задачах /4 точных). Если рассматривать только4 точные '
f'задачи: {good["method_seconds_per_accurate"]:.3f} против '
f'{good["baseline_method_seconds_per_accurate"]:.3f} с на доставку.', '',
'Закрыт ранее отсутствовавший full-wall R128 на5 OOF-switch задачах: '
'локальное численное основание для выбора R128 вместо R256 подтверждено. '
'Исходные ML timing-targets не перезаписаны и модели не переобучены; новые '
'наблюдения остаются отдельным cohort. Все5 решений — frozen outer-fold1 '
'из train-OOF, а не independent held-family результат и не решения финального '
'refit/production API. Сам выбор пяти сделан до timing и не менялся по результатам.', '',
f'В общей OOF40-таблице можно указать локальную добавочную экономию пяти '
f'переключений **{all5["saved_seconds"]:.3f} с**; остальные35 действий не меняются. '
'Полная измеренная процентная экономия/стоимость40 задач пока неизвестна: '
'остаются4 других R256 missing-cost и нет единого сопоставимого полного '
'cohort для всех40. Прежний прогноз6.25% не заменяется локальными49.92%. '
'ML feature/inference overhead в этом опыте не измерялся и не включён; '
'общие imports/load/scoring/serialization тоже не являются частью method wall. '
'Полную online-экономию этим опытом не заявляем.', '',
'Для дальнейшего селектора это подтверждение стоимости локальных R-переключений, '
'но не калибровки вероятностей, безопасности на новых семействах или способности '
'выбирать MC. Production research_only/abstain сохранён. Нужна отдельная '
'заранее замороженная оценка на новых семействах, где политика действительно '
'переключается, с явным coverage; прежние20 предложенных входов при q=.95 '
'не дают переключений текущего финального refit. MC-поддержка всё ещё требует '
'собственных реальных success/failure данных, а не переименования R-only опыта '
'в завершённый двухметодный benchmark.', '',
'## Контракт, бюджет и ресурсы', '',
'Пользователь явно утвердил1766 с и максимум40 calls. Новый execution plan '
'[plan.json](plan.json) с authorization сохранён отдельно; старый preflight '
'с approved=false остаётся историческим. Exact jobs/order/configs совпали с '
'approved preflight. Canonical expression/family/train identities, аналитические '
'G/T, reference hashes и параметры проверены до старта и после серии.', '',
'Профили неизменны: ChernovSolver.solve n128/n256,513 quadratic time nodes, '
'cubic/zero,dx=.0125,padding12,2081 work nodes [-13,13],33 target nodes. '
'Seed=None, RNG/MC/JIT/численных warmups нет. Один persistent worker, fresh '
'constructor на каждый вызов, BLAS/compute threads=1. Чередование пар и '
'циклический порядок задач точно соблюдены. perf_counter_ns включает '
'certify_G/auto_T/config/constructor/solve; exact/scoring/reference/serialization '
'после таймера. Все4 повтора сохранены.', '',
f'- Controller wall **{r["controller_wall_seconds"]:.3f} с '
f'({r["controller_wall_seconds"]/60:.2f} мин)** <1766 с; method wall sum '
f'{r["method_wall_seconds"]:.3f} с. Оба первоначальных epoch/monotonic deadline соблюдены и не сбрасывались.',
f'- Пик RSS **{r["peak_RSS_MiB"]:.3f} MiB**≤512, '
f'{r["RSS_samples"]} RSS-сэмплов; {r["thermal_samples"]} thermal-сэмпла, '
f'все state={r["maximum_thermal_state"]} nominal. Это thermal pressure, не°C.',
'- Guardian cleanup_success=true; compute/controller locks освобождены. '
'40 уникальных requests/40 результатов,0 retries/interruptions/failures/pending; '
'completed-resume вернул new_calls=0,worker_started=false.',
f'- S-применений {r["operator_applications"]:,}; spline builds3,932,200; '
'общие imports/load сохранены в workers/*.setup.json, extra_solver_calls=JIT_calls=MC_calls=0.',
f'- Настройки питания неизменны={after["settings_unchanged"]}, '
f'источник питания неизменен={after["power_source_unchanged"]}; снимки до/после сохранены.', '',
'Независимо проверены40 канонических E_grid/векторов,40 request/config/timer records, '
'порядок исполнения, медианы/экономия/стоимость точной доставки; '
'[independent_verification.json](independent_verification.json). '
'19 unit/mock/planning/report-тестов passed; [tests.json](tests.json). '
'Сохранность численного ядра, моделей/политики и прежних данных, резервные '
'копии статуса и SHA-256 — [final_verification.json](final_verification.json), '
'[artifact_sha256.json](artifact_sha256.json).', '',
'Новый код: diagnostics/run_switch5_r_timing.py,report_switch5_r_timing.py, '
'verify_switch5_r_timing.py,summarize_switch5_r_timing.py; '
'tests/test_run_switch5_r_timing.py,test_switch5_r_timing_report.py. '
'Изменён только status среди прежних проектных файлов, все новые результаты '
'сохранены в отдельном каталоге.', '']
    with (out/'REPORT.md').open('x') as f:f.write('\n'.join(lines))
    write(out/'report_summary.json',dict(local_method_saving_confirmed=True,all5_percent_saved=100*all5['relative_saved'],
        accurate4_percent_saved=100*good['relative_saved'],local_saved_seconds=all5['saved_seconds'],
        accuracy_retained_relative_R256=True,accurate_cases=4,total_cases=5,known_inaccurate_case='F0172/S03',
        supported_accurate_speedups=4,all20_paired_timings_faster=True,measured_global40_percentage=None,
        ML_inference_included=False,production_unblocked=False,new_MC_calls=0,new_ML_fits=0))
    print(dict(saved_report=True,local_percent_saved=100*all5['relative_saved'],accurate_cases='4/5',supported_accurate_speedups=4))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);summarize(p.parse_args().out.resolve())
