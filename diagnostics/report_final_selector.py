"""Scientific closure report for the finished implementation and pending final test."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ml.revision_evidence import read,sha
from diagnostics.prepare_development8 import write_json


def report(out):
    m=read(out/'training/metrics.json');r=read(out/'model/registry.json');freeze=read(out/'release_freeze.json');p=out/'final_experiment'
    b=read(p/'budget_proposal.json');plan=read(p/'plan.json');choices=read(p/'frozen_predictions.json')['choices'];name=r['final_policy']['name']
    lines=['# Итоговый научный отчёт: ML-селектор R128/R256/MC1', '', '2026-10-27, Europe/Moscow.', '',
        '**Реализация исследовательского ML-селектора завершена. Независимая финальная численная проверка подготовлена, но не запускалась и не разрешена.** Сравнение с baseline не является требованием, которое подгоняется до положительного результата. Отрицательный результат финального теста будет допустимым научным итогом.',
        '', '## Готовый результат', '',
        'Модель: model/registry.json и6 автономно экспортированных heads/scalers. Обучение воспроизведено с нуля исключительно на разрешённом train:103 R128 cases/90семей,40 R256 и40 MC1 уравнений; у MC180 уникальных seed-наблюдений и12 successes в3семействах. Повторные seed-replay не считаются новыми испытаниями, независимые seeds — k/n одного уравнения. Development8, прежняя validation/test и отложенный MC-seed audit не участвовали в fit/выборе. Предыдущие модели и данные сохранены.',
        '', 'Рабочий inference: `ml.final_selector.FinalResearchSelector.predict(coefficients,x_grid,coefficient_metadata)` и CLI `diagnostics/predict_final_selector.py`. Он возвращает R128/R256/MC1, оценки вероятности события E_grid≤.01, полные warm times, список модельно допустимых профилей и uncertainty. Требуются только заданные a,b,c,g на256точках и whitelist коэффициентных AST/сертификатов. ID/exact/error/seed/solver-output не используются. Никаких solver-вызовов внутри inference нет.',
        '', f"Замороженная train-selected политика: **{name}**, δ=.02. Среди профилей с вероятностью достижения E_grid≤.01 не более чем на.02 ниже максимальной выбирается самый быстрый по прогнозу полного времени; для MC действует10% practical tie в пользу R. **Это правило прогнозируемой допустимости, не знание будущей ошибки и не условие p≥.95.** Порог численной точности в targets неизменен:.01 на фиксированной33-точечной сетке. Даже корректно реализованный ML может выбрать неточный профиль; это предмет финальной проверки.",
        '', 'Прежний production API не изменён и остаётся blocked. Отдельный research API действительно выдаёт профиль, а не скрывает готовую модель за постоянным отказом. `production_select_method` также отказывает. Unsupported domain/profile/timing context отклоняется, повреждённые артефакты и неправильные входы проверяются. CLI успешно выполнен на input-only TRAIN-примере; это functional test, не оценка качества.',
        '', '## Train-only оценка', '',
        'Три outer family-fold; внутри каждого отдельные3 policy-fold, внутри каждого3 calibration-fold. Все ancestor-held families исключены из scaler/base/calibrator/policy fits. Заранее фиксированная небольшая сетка правил обучалась без held-out настройки. Окончательная политика refit/selected на всём train после внешней OOF оценки.312 model/scaler узлов,39calibrators;4326 внутренних OOF значений проверены независимо. Весь outer OOF и итоговые прогнозы воспроизвелись побитово относительно предыдущей train-ревизии.',
        '', '|Вариант|Точных /40|Нарушений|Отказов|Цена / точный на9 полностью измеренных задачах, с|', '|---|---:|---:|---:|---:|']
    for key in ('constant_R128','constant_R256','nested_selected',name):
        v=m['policies'][key];lines.append(f"|{key}|{v['accurate_deliveries']:g}|{v['finite_accuracy_violations']:g}|{v['abstentions']}|{v['common_complete_cost']['cost_per_accurate']:.6f}|")
    lines += ['', 'Основной честный nested-selected результат совпал с R256:33/40 точных,7нарушений,0отказов, все40действий R256. Финальная research-политика на общей train-OOF таблице переключает11R128/29R256 при33/40 точных. Это **post-selection train diagnostic**, не новая независимая оценка: выбор политики использовал эту таблицу. На9 cost-complete задачах она экономит≈22.16% противR256 при8точных/9; R128 дешевле, но точен7/9. Нельзя заявлять превосходство над обоими baseline.',
        '', 'Реальные full method-wall targets имеются только на9R128/36R256/40MC1. Стоимость всех40 случаев остаётся unknown:31R128+4R256 отсутствуют и не заполнены прогнозами. Стоимость неверных ответов включена; ML-overhead измерен warm и прибавлен как proxy. Исторические timing cohorts не объявляются новым uniform benchmark. Недостающие времена старого train не являются причиной откладывать готовую реализацию: финальный эксперимент измеряет все3 профиля на новых задачах.',
        '', 'Новые cost-heads дают семейные OOF MAPE≈2.72% R128,1.77% R256,34.55% MC. Binary R128/MC не превосходят simple outer-train constant probability; sigmoid не гарантирует улучшения proper score. Continuous-error R-head улучшает Brier, но допускает ошибки порядка. Строгий q=.95 после sigmoid подавляет R128-switches; raw q95 теряет одну точную доставку. Выбор relative-parity выполнен на train и не выдаётся за reliability=.95.',
        '', 'MC сохранён полноценным кандидатом, но в train-OOF не выбран.3positive-family недостаточно для убедительного вывода о переносе MC-надёжности. Unit-тест с mock-estimates проверяет доступность MC-ветви алгоритма, **не** создаёт научный MC-winner. Нет обязательного20+5 gate, искусственных успехов и смены численных профилей ради класса.',
        '', '## Подтверждено и не подтверждено', '',
        'Подтверждено: реализованы коэффициентный inference, воспроизводимое train-only обучение/выбор политики, семейные OOF-прогнозы вероятности/времени, корректная dataset-qualified идентичность, дедупликацияMC, экспорт/reload и блокировкаproduction. Финальная модель/код заморожены до создания новых test-входов.',
        '', 'Не подтверждено: точность/цена на новых семействах, преимущество надbaseline, гарантированное E_grid≤.01 каждого ответа, калиброванная per-input вероятность, выбор реальных MC-winner и надёжность нового MC-seed. Хвостовая аналитическая оценка не ограничивает полную ошибку MC; h-bias и sampling variance не устранены ML. Реализованный inference не является oracle и не вычисляет exact до выбора.',
        '', '## Один финальный независимый эксперимент', '',
        'Манифест: final_experiment/manifest.csv и plan.json.60 действительно новых topology families,3 на каждую из20 complexity/regime cells. Входной пул all_methods_v1 seed2488001001,800семей×1, остальные параметры генератора неизменны; используются только60 новых test-role уравнений. Это построение входов, не benchmark800/1000 и не численный пилот. Исключены все известные topology identities, включая старый test только на уровне идентичности. Отбор по фиксированному SHA256, до прогнозов, без error/time filtering и добораMC.',
        '', 'На семью4R128+4R256+4MC1: **720 вызовов максимум,240 каждого профиля**. MC-seeds248800100100..103 новые; point-addressed RNG включает ID/x. Каждая задача имеет зафиксированные AST/input hashes, общий аналитическийG/T, N, actual MC-work и resource ceilings. Seeds/профили/данные не меняются по исходам. Primary episode0, остальные3 — sensitivity; все ошибки и неудачи сохраняются, лучшего seed и M16384 из четырёх запусков не создаётся.',
        '', f"Сценарий **{b['scenario_seconds']:.3f} с (6ч52мин19.57с)**, предложенный hardcap **{b['proposed_cap_seconds']} с (13ч49мин40с)**. План:1compute worker/thread,RSS≤512MiB,guardian,external owner lease300с,thermal stop state≥2,watchdogs,immutable records и исходные epoch/monotonic deadlines. MC budget использует max(frozen-time forecast,actual-work×TRAIN median sec/update), затем2×общий сценарий+300с; это резерв, не доказанная верхняя граница.",
        '', f"Фиксированные до solver решения: **{choices}**.MC не выбран ни на одном из60 входов, и это **не причина менять политику/семейства**. Полные240 MC-вызовов дадут независимую оценку прогнозов и counterfactual-полезностиMC. Они не подтвердят conditional качество MC-selected действия при его нулевой поддержке; положительный вывод о распознавании MC-winner останется неподтверждённым, если данных нет.",
        '', 'Критерий завершения: все720 terminal outcomes учтены/проверены, frozen decisions/inputs/models неизменны, итоговые accuracy/time/refusal/cost/probability/oracle metrics и uncertainty сформированы по заранее написанному evaluator. **Не требуется превосходство ML или наличие MC-побед.** Корректный отрицательный/нулевой результат завершает научный эксперимент. При resource/data stop сохраняется partial state и фиксируется incomplete; никаких автоматических новых seed/retry/cohort/budget расширений.',
        '', 'Основные сравнения: correct deliveries/60,finite violations,отказы/сбои,полная warm стоимость иcost-per-correct относительноconstant_R128/R256; реальные fastest-accurate primary profiles с10% MC-tie.10000 paired family-bootstrap внутри20страт сохраняют seed-кластеры. Reference conservative loss bounds требуют явно заявленной независимости семейств; bootstrap/model-spread не превращаются в accuracy certificate. Все исходы анализируются один раз, модель после них не подбирается.',
        '', '**Эксперимент не запускался:**0requests,0records,нетexecution_authorization и runtime budget/deadline. Контроллер требует отдельный approval с точным plan SHA,720calls и49780с до созданияworker. Тесты подтвердили невозможность solver/worker старта без него. Авторизация не создаётся текущим запросом на подготовку.',
        '', '## Запуск и финальное действие', '',
        'Исследовательский inference готов:', '', '```bash',
        'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B diagnostics/predict_final_selector.py \\',
        '  --model results/ml_selector_final_2026-10-27/model \\',
        '  --input results/ml_selector_final_2026-10-27/example_input.json', '```', '',
        'Безопасная проверка плана без исполнения:', '', '```bash',
        'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B diagnostics/run_final_selector_experiment.py check \\',
        '  --out results/ml_selector_final_2026-10-27/final_experiment', '```', '',
        'После отдельного разрешения используется тот же frozen plan и matching authorization JSON, без перегенерации входов. Guardian lease должен продлевать внешний владелец; controller не продлевает его сам. Saved-only evaluator: diagnostics/evaluate_final_selector_experiment.py. Он сейчас возвращает not_executed, а не выдуманные метрики.',
        '', 'Реализация ML закрыта как исследовательский программный результат. Осталось **разрешить и один раз выполнить этот финальный независимый эксперимент**, проверить его исходы и зафиксировать окончательные научные выводы. Production остаётся заблокированным до отдельного подтверждения качества. Новые предварительные пилоты и дополнительное обучение не являются продолжением этого релиза.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write_json(out/'result_summary.json',dict(implementation_complete=True,research_inference_working=True,production_blocked=True,
        training_only=True,OOF_primary=dict(correct=33,requests=40,actions=dict(R256=40)),baselines=dict(constant_R128_correct=25,constant_R256_correct=33),
        final_policy=r['final_policy'],post_selection_train_diagnostic=dict(correct=33,requests=40,R128=11,R256=29),
        final_independent_experiment=dict(families=60,maximum_calls=720,proposed_cap_seconds=49780,approved=False,executed_calls=0,planned_choices=choices),
        scientific_quality_independently_confirmed=False,MC_winner_recognition_confirmed=False,superiority_required=False),exclusive=True)
    (out/'README.md').write_text('# Final research selector\n\n'+
        'Готовая модель: `model/`. Рабочий API: `ml.final_selector.FinalResearchSelector.predict`.\n\n'+
        'Научный отчёт: [REPORT.md](REPORT.md). Обучение/OOF: `training/`.\n\n'+
        'Единственный финальный тест: `final_experiment/manifest.csv`, `plan.json`, `analysis_spec.json`, `budget_proposal.json`, `approval_request.json`. Пока не разрешён и не запускался.\n\n'+
        'E_grid≤.01 — событие, предсказываемое моделью. Модельная допустимость не является гарантией точности; production API остаётся заблокированным.\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);report(ap.parse_args().out.resolve())
