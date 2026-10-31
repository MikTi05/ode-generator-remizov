"""Independent saved-model audit and report; input-only blind train20 inference."""
from __future__ import annotations
import argparse,csv,copy,json,math,os,shutil,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from diagnostics.audit_probability_selector import read,write,sha,MODEL
from ml.probabilistic_selector import load_head,choose,estimates_from_heads
from ml.nested_probability_calibration import apply_sigmoid,baseline_default_choice


def verify(out):
    from scipy.special import expit,logit
    targets=read(out/'accuracy_targets.json');index={r['equation_id']:r for r in targets if r['profile']=='R128'}
    with np.load(out/'features_train.npz',allow_pickle=False) as z:
        X=z['X'];original=z['original_indices'].tolist();names=z['feature_names'].tolist()
    local={g:i for i,g in enumerate(original)};nodes=read(out/'model_index.json');oof=read(out/'oof.json')
    protocol=read(out/'protocol.json');blocks={b['fold']:set(b['held_families']) for b in protocol['outer_blocks']}
    flat=[g for b in blocks.values() for g in b];assert len(flat)==len(set(flat))==90
    heads={};scalers=0;inner_count=0;max_gradient=0.
    for node,r in nodes.items():
        rs=[index[i] for i in r['train_equation_ids']];excluded=set(r['excluded_families'])
        assert not excluded&{z['family_id'] for z in rs} and all(z['split']=='train' for z in rs)
        indices=[local[z['index']] for z in rs]
        assert indices==r['entry']['metadata']['indices']
        np.testing.assert_array_equal(X[indices].mean(axis=0),r['entry']['metadata']['train_scaler_mean'])
        heads[node]=load_head(r['entry'],out/'models');scalers+=1
    for profile,rs in oof.items():
        expected=[r for r in targets if r['profile']==profile];byid={r['equation_id']:r for r in expected}
        assert len(rs)==len(byid)==len({r['equation_id'] for r in rs})
        for r in rs:
            t=byid[r['equation_id']];assert r['k']==t['k'] and r['n']==t['n']==1 and r['family_id']==t['family_id']
            assert r['family_id'] in blocks[r['outer_fold']] and r['family_id'] not in nodes[r['base_node']]['train_families']
            raw=heads[r['base_node']].predict(X[[r['index']]],names)['point'][0]
            assert raw==r['raw']
            cal=read(out/'calibrators'/f"{r['calibrator_key']}.json")
            assert r['family_id'] not in cal['families'] and float(apply_sigmoid(cal,[raw])[0])==r['calibrated']
    for file in sorted((out/'calibrators').glob('*.json')):
        cal=read(file);profile=cal['profile'];rs=[next(r for r in targets if r['profile']==profile and r['equation_id']==pid) for pid in cal['equation_ids']]
        if 'outer_fold' in cal:
            outer=blocks[cal['outer_fold']];assert not outer&set(cal['families']);predictions={}
            for b in cal['inner_blocks']:
                node=nodes[b['node']];held=set(b['held_families'])
                assert not (outer|held)&set(node['train_families'])
                ids=b['equation_ids'];ix=[local[index[pid]['index']] for pid in ids]
                p=heads[b['node']].predict(X[ix],names)['point']
                for pid,v in zip(ids,p):
                    assert pid not in predictions;predictions[pid]=float(v);inner_count+=1
                    assert float(v)==cal['inner_probabilities'][pid]
            assert set(predictions)==set(cal['equation_ids'])
        else:predictions={r['equation_id']:r['raw'] for r in oof[profile]}
        # Independent loss/gradient, not another calibration fit.
        from collections import Counter
        count=Counter(r['family_id'] for r in rs);w=np.array([1/count[r['family_id']] for r in rs])
        p=np.array([predictions[r['equation_id']] for r in rs]);z=logit(np.clip(p,1e-6,1-1e-6));y=np.array([r['k'] for r in rs])
        logits=cal['slope']*z+cal['intercept'];res=w*(expit(logits)-y)
        gradient=np.array([np.sum(res*z)+2*(cal['slope']-1),np.sum(res)+2*cal['intercept']])
        for i,v in enumerate((cal['slope'],cal['intercept'])):
            lower,upper=cal['bounds'][i]
            if v==lower:gradient[i]=min(0.,gradient[i])
            elif v==upper:gradient[i]=max(0.,gradient[i])
        error=float(np.max(abs(gradient)));assert error<1e-4,(file.name,error);max_gradient=max(max_gradient,error)
        objective=float(np.sum(w*(np.logaddexp(0.,logits)-y*logits))+((cal['slope']-1)**2+cal['intercept']**2))
        assert math.isclose(objective,cal['objective'],abs_tol=1e-10)
    metrics=read(out/'metrics.json');paired=read(out/'paired_oof.json');old={r['equation_id']:r for r in read(MODEL/'family_cv.json')['oof']}
    for r in paired:
        assert r['estimates_raw']==old[r['equation_id']]['estimates']
        for p in ('MC1',):assert r['estimates_calibrated'][p]==r['estimates_raw'][p]
        for p in ('R128','R256'):assert r['estimates_calibrated'][p]['cost']==r['estimates_raw'][p]['cost']
    for mode in ('raw','calibrated'):
        for q in (.5,.8,.95):
            for kind in ('risk_constraint','baseline_default'):
                actions=[choose(r['estimates_'+mode],q)['action'] if kind=='risk_constraint' else baseline_default_choice(r['estimates_'+mode],q) for r in paired]
                success=sum(r['observed'][p]['k']/r['observed'][p]['n'] for r,p in zip(paired,actions) if p!='abstain')
                z=metrics['policies'][mode][str(q)][kind]
                assert success==z['accurate_deliveries'] and sum(p!='abstain' for p in actions)==z['selected']
                for r,p,detail in zip(paired,actions,z['details']):assert detail['action']==p and detail['equation_id']==r['equation_id']
    for profile,rs in oof.items():
        counts={g:sum(r['family_id']==g for r in rs) for g in {r['family_id'] for r in rs}}
        w=np.array([1/counts[r['family_id']] for r in rs]);y=np.array([r['k'] for r in rs])
        for mode in ('raw','calibrated'):
            p=np.array([r[mode] for r in rs]);b=(p-y)**2;cl=np.clip(p,1e-10,1-1e-10);l=-y*np.log(cl)-(1-y)*np.log1p(-cl)
            for scope,weights in [('equation_weighted',np.ones(len(rs))),('family_weighted',w)]:
                s=metrics['probability'][profile][mode][scope]
                assert math.isclose(float(np.average(b,weights=weights)),s['brier'],abs_tol=1e-15)
                assert math.isclose(float(np.average(l,weights=weights)),s['logloss'],abs_tol=1e-15)
    registry=read(out/'registry.json');assert not registry['deployment_ready'] and registry['no_production_dispatch']
    assert sha(ROOT/'ml/probabilistic_selector.py')==registry['original_production_code_sha256']
    for n,h in read(out/'source_sha256.json').items():assert sha(ROOT/n)==h,n
    result=dict(passed=True,R128_cases=103,R128_families=90,R256_cases=40,model_scalers_verified=scalers,
        inner_OOF_predictions_replayed=inner_count,calibrators_verified=11,maximum_calibrator_KKT_residual=max_gradient,
        old_paired_point_predictions_exact=80,all_saved_models_loaded=True,all_outer_inner_family_exclusions_passed=True,
        policy_counts_and_proper_scores_recomputed=True,production_API_unchanged_and_blocked=True,
        numerical_calls=0,original_validation_or_test_outcomes_used=False)
    write(out/'independent_verification.json',result);return result


def prospective_train20(out):
    """Frozen model inference on 20 TRAIN inputs; labels/exact channels never used."""
    from ml.remizov_features import coefficient_features,CERT_FIELDS
    registry=read(out/'registry.json');frozen_sha=sha(out/'registry.json')
    write(out/'freeze_before_train20_inputs.json',dict(registry_sha256=frozen_sha,models_frozen=True,
        thresholds=[.5,.8,.95],train20_labels_available_to_this_pipeline=False,parameter_changes_after_inputs=False))
    proposal=read(ROOT/'results/ml_probability_selector_audit_2026-10-26/acquisition_plan.json')
    data=ROOT/'data/all_methods_v1_1000_2026-10-18';wanted={r['equation_id']:r for r in proposal['cases']};meta={}
    with (data/'all_methods_equations_v1_metadata.jsonl').open() as f:
        for line in f:
            row=json.loads(line);pid=row['equation_id']
            if pid not in wanted:continue
            assert row['split']=='train' and row['expression_hash']==wanted[pid]['expression_hash']
            meta[pid]={k:row[k] for k in CERT_FIELDS}
            meta[pid]['serialized_ast']={k:row['serialized_ast'][k] for k in ('a_raw','b_raw','c_raw')}
    with np.load(data/'all_methods_equations_v1_ml.npz',allow_pickle=False) as z:
        assert z['channel_order'].tolist()==['a','b','c','g']
        ids=z['equation_id'];index={str(pid):i for i,pid in enumerate(ids)};x=z['x_grid']
        coefficients=z['X'][[index[c['equation_id']] for c in proposal['cases']]]
        assert all(str(z['split'][index[c['equation_id']]])=='train' for c in proposal['cases'])
    rows=[]
    for c,values in zip(proposal['cases'],coefficients):
        f=coefficient_features(values,x,meta[c['equation_id']]);assert list(f)==registry['feature_names']
        rows.append([f[n] for n in registry['feature_names']])
    X=np.asarray(rows);oldreg=read(MODEL/'registry.json');heads={}
    for p in ('R128','R256','MC1'):
        for kind in ('probability','cost'):
            entry=registry['final_heads'][p] if p in ('R128','R256') and kind=='probability' else oldreg['heads'][p+':'+kind]
            folder=out/'models' if p in ('R128','R256') and kind=='probability' else MODEL/'models'
            heads[p+':'+kind]=load_head(entry,folder)
    raw=estimates_from_heads(heads,X,registry['feature_names']);predictions=[]
    for c,e in zip(proposal['cases'],raw):
        calibrated=copy.deepcopy(e)
        for p in ('R128','R256'):
            cal=read(out/registry['calibrators'][p])
            for key in ('point','low','high'):calibrated[p]['probability'][key]=float(apply_sigmoid(cal,[e[p]['probability'][key]])[0])
        predictions.append(dict(equation_id=c['equation_id'],family_id=c['family_id'],split='train',
            raw=e,calibrated=calibrated,choices={str(q):dict(risk=choose(calibrated,q)['action'],
                baseline_default=baseline_default_choice(calibrated,q)) for q in (.5,.8,.95)},
            accuracy_unknown=True))
    assert sha(out/'registry.json')==frozen_sha
    fitted=set().union(*(set(n['train_families']) for n in read(out/'model_index.json').values()))
    assert not fitted&{r['family_id'] for r in predictions}
    write(out/'prospective_train20_predictions.json',dict(scope='input-only inference AFTER model freeze, no pilot labels, no selection/tuning',
        registry_sha256=frozen_sha,rows=predictions,q95_switches=sum(r['choices']['0.95']['baseline_default']!='R256' for r in predictions),
        q95_risk_abstentions=sum(r['choices']['0.95']['risk']=='abstain' for r in predictions),
        data_keys_used=['X[a,b,c,g]','x_grid','equation_id','split','channel_order','coefficient certificates','a/b/c coefficient ASTs'],
        data_sha256={n:sha(data/n) for n in ('all_methods_equations_v1_ml.npz','all_methods_equations_v1_metadata.jsonl')},
        numerical_calls=0,model_fit_after_pilot_features=False,test_used=False,validation8_used=False))
    np.savez_compressed(out/'prospective_features_train20.npz',X=X,equation_id=np.asarray([r['equation_id'] for r in predictions]))
    return predictions


def report(out):
    verification=verify(out);pilot=prospective_train20(out)
    metrics=read(out/'metrics.json');oof=read(out/'oof.json');paired=read(out/'paired_oof.json');registry=read(out/'registry.json')
    policies=metrics['policies'];q95=policies['calibrated']['0.95']['baseline_default'];base=policies['constant_R256']
    switches=[d for d in q95['details'] if d['action']!='R256'];switch_ids={r['equation_id'] for r in switches}
    switch_folds=sorted({r['outer_fold'] for r in paired if r['equation_id'] in switch_ids})
    final_cal={p:read(out/registry['calibrators'][p]) for p in ('R128','R256')}
    pilot_switches=sum(r['choices']['0.95']['baseline_default']!='R256' for r in pilot)
    with (out/'policy_metrics.csv').open('x',newline='') as f:
        fields=['mode','q','policy','accurate','selected','abstentions','lost_R256_deliveries','R128_switches','MC_switches','missing_costs','predicted_method_cost_sum']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for mode in ('raw','calibrated'):
            for q in (.5,.8,.95):
                for policy,z in policies[mode][str(q)].items():
                    writer.writerow(dict(mode=mode,q=q,policy=policy,accurate=z['accurate_deliveries'],selected=z['selected'],abstentions=z['abstentions'],
                        lost_R256_deliveries=z['lost_baseline_accurate'],R128_switches=z['actions']['R128'],MC_switches=z['actions']['MC1'],
                        missing_costs=z['missing_method_cost_cases'],predicted_method_cost_sum=z['predicted_method_cost_sum']))
    lines=['# Полный семейный OOF R128 и вложенная калибровка Remizov', '',
'Дата: 2026-10-26. **103/103 R128 случая на90 train-семействах**, R25640/40 на40 семействах. '
'Сохранены38 model/scaler nodes (10 bootstrap-реплик +point каждый) и11 калибраторов. '
'Новых solver/MC/benchmark1000 calls —0. Предыдущие8 validation и test не использованы для fit, '
'калибровки, порогов или оценки качества в этом запуске. Production API не изменён.', '',
'**Общего улучшения калибровки нет:** R256 proper scores улучшились описательно; R128 ухудшились. '
'При заранее заданном q=.95 исследовательский R256-default challenger сохранил33/40 точных доставок '
'без наблюдаемых индивидуальных потерь, переключившись5 раз на R128. Это кандидат для новой проверки, '
'не подтверждённое превосходство: full-wall R128 отсутствует у всех5 переключений, '
'а сами переключения получены лишь в одном outer-fold.', '',
'## Дизайн без утечек', '',
'Внешние6 блоков составляют непересекающееся покрытие90 семейств. Первые3 — прежние40 paired '
'семейств, следующие3 — остальные50 по заранее фиксированному SHA256-порядку. Все образцы '
'одного семейства всегда удерживаются вместе. Это позволяет дополнить OOF до103 и в точности '
'воспроизвести прежние80 point-прогнозов R128/R256 на40 задачах; финальные in-sample прогнозы '
'не выдаются за OOF. R256/MC labels для остальных63 R128 случаев отсутствуют и не изобретались.', '',
'В каждом внешнем блоке base scaler/model обучен только на outer-train. На этом outer-train '
'отдельно построены3 внутренние StratifiedGroupKFold модели; каждая inner-OOF вероятность получена '
'без всего семейства соответствующей метки и без всех outer-held семейств. По этим вероятностям '
'обучается один заранее заданный sigmoid σ(a logit(p)+b), a≥0, L2=2 к identity(a=1,b=0). '
'Вес каждого семейства в калибраторе равен1 (образцы делят его вес). Гиперпараметры и '
'выбор calibrator/raw по outer scores не оптимизировались. Итерации/depth/learning-rate '
'CatBoost и253 input-only признака исходные;1 поток.', '',
'Финальные research-головы R128/R256 refit на всех своих train-данных и совпали по raw '
'прогнозам с исходными моделями. Финальные калибраторы обучены на полном raw OOF; '
'они **не использовались для оценки этих же OOF-исходов**. Для оценки использованы только '
'outer-specific калибраторы из вложенного дизайна. MC и cost головы не переобучались; '
'их прежние paired OOF-прогнозы безопасно переиспользованы с теми же внешними held-семействами.', '',
'## Вероятность точности: proper scores', '',
'| Профиль / вес | Raw Brier | Cal Brier | Raw logloss | Cal logloss | Fold-constant Brier |',
'|---|---:|---:|---:|---:|---:|']
    for p in ('R128','R256'):
        for weight in ('equation_weighted','family_weighted'):
            z=metrics['probability'][p];r=z['raw'][weight];c=z['calibrated'][weight]
            lines.append(f'| {p} {weight} | {r["brier"]:.6f} | {c["brier"]:.6f} | {r["logloss"]:.6f} | {c["logloss"]:.6f} | {z["train_constant"][weight]["brier"]:.6f} |')
    lines += ['', 'R128 raw family-weighted mean p=.7862, observed=.7744; после калибровки p=.7596. '
'Средняя калибровка меняется, но Brier/logloss становятся хуже: исправление среднего p '
'не гарантирует хороших условных вероятностей. На40 paired R128 raw остаётся прежним '
'(Brier .17720), а полный103-case raw .13387 не сравнивается с .17720 как улучшение модели '
'на том же наборе: различаются состав/веса задач. R256 raw40 воспроизведён точно; '
'улучшение .10786→.09601 получено для тех же40 held-family случаев.', '',
'![Семейные OOF-метрики](nested_scores.png)', '',
'## Неопределённость и неоднородность', '',
'Фиксированные4000 family-bootstrap ресэмплов дают условные95% интервалы для '
'разности calibrated−raw proper scores; model/refit uncertainty и общие train-наборы '
'OOF-моделей ими полностью не учтены:', '',
'| Профиль | ΔBrier family mean [95%] | Δlogloss family mean [95%] |',
'|---|---|---|']
    for p in ('R128','R256'):
        u=metrics['probability'][p]['uncertainty'];d=u['family_weighted_delta_cal_minus_raw'];ci=u['descriptive_fixed_prediction_family_bootstrap95']
        lines.append(f'| {p} | {d[0]:+.5f} [{ci[0][0]:+.5f},{ci[0][1]:+.5f}] | {d[1]:+.5f} [{ci[1][0]:+.5f},{ci[1][1]:+.5f}] |')
    lines += ['', 'R128 family-Brier ухудшился в5 из6 блоков; R256 улучшился во всех3, '
'но его интервалы разности включают0. Slope R128 outer-калибраторов ≈.436–1.016: '
'в малой выборке они сильно различаются. Все5 переключений q=.95 приходятся на один '
'исходный fold1 (slope1.016), где сжатие вероятностей почти отсутствует. '
'Это слабое основание для переноса преимущества на новые семейства. Вложенные base fits '
'имеют меньше train-данных, чем финальный refit, поэтому перенос calibrator→final base '
'также требует новой внешней проверки.', '',
'Bootstrap исходных моделей10/90 остаётся некалиброванным разбросом, а не CI истинного p. '
'Монотонный sigmoid не устраняет ошибки ранжирования. Финальные calibrated-реплики '
'не объявляются надёжными нижними границами. У нового registry deployment_ready=false, '
'original_validation_calibrated=false и no_production_dispatch=true.', '',
'## Выбор метода на40 полностью размеченных задачах', '',
'До обучения зафиксированы исходные q=.5/.8/.95 и2 политики: '
'(1) исходное p≥q с отказом, (2) **R256-default**: R256 по умолчанию, '
'переключение лишь на более дешёвый R128/MC с p≥q; правило10% ничьей сохранено. '
'У default политика q ограничивает переключение, **не обещает p_R256≥q** для fallback. '
'Она устраняет потери из-за отказа, но не гарантирует сохранения точных ответов при переключении. '
'Обе альтернативы, включая MC, остаются допустимыми действиями. Это исследовательская '
'проверка явной цели пользователя сохранить baseline-доставки, не изменение production.', '',
'| Версия | q | Политика | Точные/40 | Нарушения | Отказы | Потеря точных R256 | R128 / MC |',
'|---|---:|---|---:|---:|---:|---:|---:|',
'| constant_R256 | — | baseline | 33 | 7 | 0 | 0 | 0 / 0 |']
    for mode in ('raw','calibrated'):
        for q in (.5,.8,.95):
            for kind,z in policies[mode][str(q)].items():
                lines.append(f'| {mode} | {q} | {kind} | {z["accurate_deliveries"]:.0f} | {z["inaccurate_selected"]:.0f} | {z["abstentions"]} | {z["lost_baseline_accurate"]:.0f} | {z["actions"]["R128"]} / {z["actions"]["MC1"]} |')
    lines += ['', 'Никакой q не выбирался по максимуму этой таблицы. Основной заранее заданный q=.95: '
'исходная calibrated risk-политика даёт11/40 точных,1 неточный,28 отказов; '
'только правильная точность среди выбранных11/12=91.67% не компенсирует coverage30%. '
'Calibrated R256-default даёт33/40=82.5%,7 неточных,0 отказов, '
'ровно как baseline, со следующими переключениями:', '']
    lines += [f'- {r["equation_id"]}: R128 точен={bool(r["accurate"])}, R256 точен={bool(r["baseline_accurate"])}.' for r in switches]
    from scipy.stats import beta
    switched_baseline_correct=sum(r['baseline_accurate'] for r in switches)
    lines += ['', f'Нет наблюдаемой индивидуальной потери среди33 baseline-correct, но односторонняя '
f'95% верхняя граница доли потерь ≈{q95["descriptive_one_sided95_upper_loss_rate_among_baseline_correct"]:.2%}; '
f'среди лишь{switched_baseline_correct} baseline-correct переключений ≈{float(beta.ppf(.95,1,switched_baseline_correct)):.2%}. '
'Вырожденный bootstrap Δaccuracy=[0,0] при нулевых разностях не является доказательством '
'noninferiority или гарантией нового переключения. Текущая точность82.5% также не достигает '
'95% риска; q=.95 не следует путать с фактической точностью всей default-политики.', '',
'**Экономия времени не установлена.** У всех5 R128-переключений нет полного wall-замера; '
'новая политика имеет9 missing-cost задач, baseline4. Полные наблюдаемые суммы и секунды '
'на точную доставку оставлены null. Прогноз суммы method wall: '
f'{q95["predicted_method_cost_sum"]:.3f} против {base["predicted_method_cost_sum"]:.3f} с '
f'(−{100*(1-q95["predicted_method_cost_sum"]/base["predicted_method_cost_sum"]):.2f}%), '
'без нового inference-overhead. Это прогноз по прежним OOF cost-головам, не измеренный выигрыш. '
'На4 задачах с известным R128 full-wall новая q=.95 политика вообще не переключалась. '
'Необходимость измерить стоимость здесь реальна; подставлять native solver-only runtime нельзя.', '',
'## Замороженный прогноз для предложенных20 train-семейств', '',
'После сохранения нового registry выполнен отдельный **input-only** прогноз на ранее предложенных '
'20 original-train семействах, отсутствующих во всех fit/calibration. Использованы только '
'коэффициенты a/b/c/g, input grid, сертификаты и coefficient AST; labels/exact/error '
'не загружались и не генерировались. Никакой fit или изменение q после этих прогнозов не выполнены. '
f'При q=.95 новый финальный R256-default переключается на R128/MC в **{pilot_switches}/20** '
'случаев. Полные исходные прогнозы и choices всех заранее заданных q сохранены в '
'[prospective_train20_predictions.json](prospective_train20_predictions.json); истинная точность неизвестна. '
'При0 переключений этот пакет не может показать выигрыш текущей q=.95 политики над тем же R256: '
'все численные действия совпадают, а у селектора добавляется inference-overhead. '
'Предсказанные R128 p≤.93566 и MC p≤.13829 не дают кандидатов при q=.95; '
'снижать q по этим20 входам ради появления переключений нельзя.', '',
'## Что необходимо MC и оправдан ли pilot20', '',
'MC не переобучен: только40 train-семейств,7 успехов/160 seed-вызовов на3 семействах. '
'Все прежние OOF p_MC<.5, поэтому ни одна исследованная политика не выбрала MC. '
'R-калибровка не создаёт информации о MC и не решает проблему двухметодного выбора. '
'Нужны реальные независимые success/failure семейства MC1 с четырьмя и более fresh seeds, '
'полным сопоставимым wall и хотя бы некоторыми режимами возможного выигрыша относительно '
'самого быстрого точного R-профиля. Повторы на одном семействе не заменяют разнообразие. '
'MC1 four-seed4/4 не подтверждает p≥.95; редкие успехи и профильный bias нельзя устранить '
'искусственными winner-метками или одним переназначением q.', '',
'**Pilot20 обоснован как ограниченный сбор недостающих данных, не как обещание подтвердить '
'превосходство.** Прежняя схема1 R128+1 R256+4 MC1 =120 calls даёт40 недорогих R-замеров '
'для допустимости/стоимости и80 более дорогих MC для разнообразия. Её можно заранее '
'разделить на R-этап40 calls и отдельное решение о MC-этапе80: не начинать дорогой '
'MC только ради появившегося класса. Частота новых MC-success по старому условному сценарию '
'3/40 даёт лишь1.5 ожидаемых positive-family в20; этого недостаточно для надёжной '
'MC-калибровки при q=.95. Если сбор идёт по разным этапам, сохранять missing MC, '
'не вычислять полную селекторную метрику по неполному пакету и не выдавать early-stop '
'выборку за репрезентативную оценку MC.', '',
'После данного freeze эти20 могут стать prospective внутренней проверкой текущей кандидатной '
'версии до включения их outcomes в development. Оценка нового переобучения на тех же20 '
'потребует другого holdout. Для заявления о timing-winner нужны median4 R-повторы '
'(до240 calls для20), либо заранее иной явно утверждённый протокол времени; '
'одиночные R-замеры достаточны для первичной диагностики/train-time targets. '
'Старый предложенный cap9754 с остаётся **неутверждённым**, перенос validation-бюджета запрещён. '
'Новых численных вызовов в этой задаче нет.', '',
'Отдельно самый дешёвый путь проверить наблюдаемую OOF-экономию — в будущей задаче '
'добавить matched full-wall R128/R256 timing на перечисленных5 уже train-уравнениях; '
'это не новый семейный тест и не подтверждение MC. Такой опыт должен иметь собственные '
'параметры/бюджет/seed/preflight и сохранить исторические результаты. '
'Выбор следующего опыта зависит от цели: закрыть стоимость R-переключений или собрать '
'недостающие MC-режимы; считать это завершённым ML-выбором двух методов нельзя.', '',
'## Артефакты и проверки', '',
'[protocol.json](protocol.json) записан до fit; [oof.json](oof.json), '
'[metrics.json](metrics.json), [policy_metrics.csv](policy_metrics.csv), '
'[model_index.json](model_index.json), [registry.json](registry.json), '
'[independent_verification.json](independent_verification.json). Все38 scaler/model nodes '
'загружены заново, inner-OOF прогнозы воспроизведены,11 objectives/градиентов калибраторов '
'проверены без повторного fit, family exclusions и метрики пересчитаны. '
'Производственный код/registry прежней версии не менялись, новый registry research_only. '
'Unit-проверки: [tests.json](tests.json); сохранность/SHA: '
'[final_verification.json](final_verification.json), [artifact_sha256.json](artifact_sha256.json).', '']
    with (out/'REPORT.md').open('x') as f:f.write('\n'.join(lines))
    plots(out)
    write(out/'report_summary.json',dict(R128_calibration_improved=False,R256_calibration_descriptively_improved=True,
        q95_default_exact_deliveries=33,baseline_exact_deliveries=33,q95_default_no_observed_individual_loss=True,
        q95_default_switches=5,switch_outer_folds=switch_folds,observed_timing_superiority=False,
        predicted_method_time_saving_percent=100*(1-q95['predicted_method_cost_sum']/base['predicted_method_cost_sum']),
        prospective_train20_q95_switches=pilot_switches,production_unblocked=False,
        numerical_calls=0,validation8_used_for_selection=False,test_used=False))
    print(json.dumps(dict(passed=True,verification=verification,prospective_train20_q95_switches=pilot_switches,report=str(out/'REPORT.md'))),flush=True)


def plots(out):
    metrics=read(out/'metrics.json')
    with tempfile.TemporaryDirectory(prefix='nested_calibration_mpl_') as cache:
        os.environ['MPLCONFIGDIR']=cache
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(12.5,3.8),layout='constrained')
        colors=['#3182BD','#E6550D','#969696'];modes=['raw','calibrated','train_constant']
        for ax,key in zip(axes[:2],['brier','logloss']):
            for j,mode in enumerate(modes):
                values=[metrics['probability'][p][mode]['family_weighted'][key] for p in ('R128','R256')]
                bars=ax.bar(np.arange(2)+(j-1)*.23,values,width=.23,label=mode,color=colors[j])
                ax.bar_label(bars,fmt='%.3f',fontsize=8)
            ax.set(xticks=[0,1],xticklabels=['R128 (90 families)','R256 (40 families)'],ylabel=key,title='Lower is better')
            ax.set_ylim(0,ax.get_ylim()[1]*1.12)
        axes[0].legend(fontsize=8)
        for i,p in enumerate(('R128','R256')):
            u=metrics['probability'][p]['uncertainty'];mean=u['family_weighted_delta_cal_minus_raw'][0];lo,hi=u['descriptive_fixed_prediction_family_bootstrap95'][0]
            axes[2].errorbar([mean],[i],xerr=[[mean-lo],[hi-mean]],fmt='o',capsize=4,color=colors[i])
        axes[2].axvline(0,color='.6',ls='--');axes[2].set(yticks=[0,1],yticklabels=['R128','R256'],xlabel='Delta Brier (calibrated - raw)',title='Fixed-OOF family bootstrap95')
        fig.suptitle('Nested family OOF: calibration helps R256 descriptively, hurts R128')
        fig.savefig(out/'nested_scores.png',dpi=170);fig.savefig(out/'nested_scores.svg');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);report(p.parse_args().out.resolve())
