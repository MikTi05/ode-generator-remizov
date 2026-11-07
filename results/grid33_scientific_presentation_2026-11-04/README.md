# Научное оформление E33 · проект №2488

Основной раздел: [section.md](section.md), LaTeX-вставка [section.tex](section.tex).
Сравнительные таблицы: [comparison.md](comparison.md), [comparison.csv](comparison.csv)
и две готовые LaTeX-таблицы. [REQUIREMENTS.md](REQUIREMENTS.md) разделяет
закрытые требования, частичное покрытие и оставшиеся эксперименты.
В figures/ находятся четыре рисунка в PNG и векторном PDF; подписи и SHA
записаны в figure_manifest.json, точные данные — в figure_data.json.
remizov_exceedances.csv содержит все 27 превышений допуска; missing_remizov.csv — три пропуска.

Новый каталог не заменяет исторический отчёт. В нём section «Отдельная проверка…»
может быть заменён этой вставкой после редакторского решения. Для Overleaf
скопируйте данный каталог как grid33_supplement/ рядом с прежним report.tex,
затем используйте `\input{grid33_supplement/section.tex}` в нужном месте.
Фигуры/таблицы используют явно этот относительный префикс и уникальные метки.
Пакеты booktabs/graphicx/русские шрифты уже есть в прежнем отчёте.
TeX-компиляция и пагинация здесь не проверены: локального движка нет.
Просмотрены готовые PNG; PDF рисунков экспортирован Matplotlib.

Воспроизведение из корня в НОВЫЙ каталог:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B diagnostics/present_saved_grid33.py --out results/grid33_scientific_reproduction
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B diagnostics/verify_grid33_presentation.py --out results/grid33_scientific_reproduction
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_grid33_presentation.py tests/test_saved_grid33.py tests/test_four_method_accuracy.py
```

Числовая/структурная проверка нового комплекта: verification.json.
При подготовке поставленного комплекта тесты, воспроизводимость и сохранность
прежних файлов дополнительно закреплены в checks.json и integrity.json.
Новых solver, MC, fit, model inference и подборов: 0.
