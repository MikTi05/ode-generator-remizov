"""Saved-only class/threshold decomposition and bounded, unapproved next plans."""
from __future__ import annotations
import argparse,math,sys,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from diagnostics.audit_probability_selector import read,write,MODEL
from ml.nested_probability_calibration import metric_summary


def run(out):
    oof=read(out/'oof.json');metrics=read(out/'metrics.json');summary={}
    for p,rows in oof.items():
        own=dict(thresholds={},class_loss_decomposition={})
        for mode in ('raw','calibrated'):
            own['thresholds'][mode]={}
            for q in (.5,.8,.95):
                selected=[r for r in rows if r[mode]>=q]
                own['thresholds'][mode][str(q)]=dict(selected_equations=len(selected),
                    selected_families=len({r['family_id'] for r in selected}),correct=sum(r['k'] for r in selected),
                    conditional_accuracy=sum(r['k'] for r in selected)/len(selected) if selected else None,
                    family_metrics=metric_summary(selected,[r[mode] for r in selected]) if selected else None,
                    all_equation_delivery_rate=sum(r['k'] for r in selected)/len(rows))
        for label in (0,1):
            own['class_loss_decomposition'][str(label)]={mode:metric_summary(
                [r for r in rows if r['k']==label],[r[mode] for r in rows if r['k']==label])
                for mode in ('raw','calibrated')}
        summary[p]=own
    write(out/'thresholds_and_mechanism.json',summary)
    candidate=metrics['policies']['calibrated']['0.95']['baseline_default']
    ids=[r['equation_id'] for r in candidate['details'] if r['action']=='R128']
    costs=read(MODEL/'full_cost_targets.json');med={p:statistics.median(r['seconds'] for r in costs if r['profile']==p) for p in ('R128','R256')}
    scenario=len(ids)*4*sum(med.values())
    nextsteps=dict(execution_authorized=False,budgets_approved=False,
        interpretation='R256 calibrated proper scores improve descriptively; R128 sigmoid is not an improvement; no production selection is made',
        immediate_research_option=dict(scope='matched timing only on5 OOF-switch TRAIN equations; not a new-family accuracy assessment',
            equation_ids=ids,calls_per_equation=dict(R128=4,R256=4,MC1=0),maximum_calls=40,
            timing='unchanged contract full warm method wall; one environment/worker/thread, RSS512, guardian/thermal; separate preflight required',
            central_scenario_seconds=scenario,proposed_cap_seconds=math.ceil(2*scenario+300),approved=False,
            seed='R profiles deterministic; no MC seeds or new MC trials',
            rationale='all5 switched R128 full times missing; match both profiles rather than pool unequal timing cohorts',
            restrictions='preserve inaccurate numerical results; no claim of population accuracy or two-method MC superiority'),
        original_train20_pilot=dict(maximum_original_calls=120,R_calls=40,MC_calls=80,budget_approved=False,
            as_superiority_test_q95=False,reason='all20 frozen final default-policy actions are R256; no switch effect to measure',
            as_data_acquisition='bounded prospective p/cost diagnostic is useful; MC80 needs separate rationale/budget; original20 then become development only after the frozen audit',
            do_not='lower q using these input forecasts, relabel old8 validation as independent, or extend sampling until MC succeeds'),
        MC_needed=dict(current_families=40,current_positive_families=3,current_seed_successes=7,current_seed_trials=160,
            data='new independent TRAIN families with real MC1 success/failure, fresh seeds, full-grid33 errors and matched full wall against fastest accurate R128/R256',
            sampling='include coverage controls plus prespecified promising MC-cost/accuracy regimes; record selection probabilities/strata and do not calibrate population rates on unweighted enriched successes',
            reliability='more seeds estimate fixed MC1 new-call p; they do not raise M or improve a subsequent single MC1 call; four trials cannot establish p>=.95',
            stop='no manufactured wins, no use of20+5 winner gate as calibration sufficiency, no automatic expensive expansion',
            scientific_goal='both Remizov and MC remain actions; R-only timing check does not complete the two-method selector'))
    write(out/'NEXT_STEPS.json',nextsteps)
    lines=['# Пороговые и классовые диагностики', '',
'Все данные — внешние train-OOF, калибраторы соответствующего outer-fold. Пороги не подбирались.', '',
'| Профиль | Версия | q | Выбрано | Точно | Условная точность | Доставок / все случаи |',
'|---|---|---:|---:|---:|---:|---:|']
    for p,own in summary.items():
        for mode,tails in own['thresholds'].items():
            for q,z in tails.items():
                ca=f'{z["conditional_accuracy"]:.2%}' if z['conditional_accuracy'] is not None else 'не определено'
                lines.append(f'| {p} | {mode} | {q} | {z["selected_equations"]} | {z["correct"]} | {ca} | {z["all_equation_delivery_rate"]:.2%} |')
    lines += ['', 'Brier по настоящему классу (equation-weighted); это описание эффекта преобразования, не выбор калибратора:', '',
'| Профиль | Истинная допустимость | Случаев | Raw Brier | Cal Brier |',
'|---|---:|---:|---:|---:|']
    for p,own in summary.items():
        for label,z in own['class_loss_decomposition'].items():
            lines.append(f'| {p} | {label} | {z["raw"]["equations"]} | {z["raw"]["equation_weighted"]["brier"]:.6f} | {z["calibrated"]["equation_weighted"]["brier"]:.6f} |')
    lines += ['', f'Минимальный предложенный matched timing-блок для5 уже train-переключений:40 R-вызовов, '
f'сценарий{scenario:.1f} с, отдельный cap{nextsteps["immediate_research_option"]["proposed_cap_seconds"]} с '
'**не утверждён**, запусков нет. Это закрытие стоимости, не независимая оценка точности и не подтверждение MC.', '',
'Исходный120-call pilot20 не способен показать преимущество текущей q=.95 политики: все20 действий совпадают с R256. '
'Для новых данных его R40-часть полезна; дорогая MC80-часть требует отдельного обоснования режима/новизны. '
'Полный план и ограничения сохранены в [NEXT_STEPS.json](NEXT_STEPS.json).', '']
    with (out/'SUPPLEMENT.md').open('x') as f:f.write('\n'.join(lines))
    print(dict(saved_only=True,next_timing_calls=40,proposed_unapproved_cap=nextsteps['immediate_research_option']['proposed_cap_seconds']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);run(p.parse_args().out.resolve())
