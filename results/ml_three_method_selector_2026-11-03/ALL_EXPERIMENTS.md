# Все заранее зафиксированные модели и политики

RK4 исключён. Основная политика определена selection_lock.json до Test; все прочие Test-результаты диагностические, переотбора по ним нет. Original — sensitivity, latest — основной сценарий.

Обучаемых основных спецификаций 96; основных estimator fit120; CV estimator fit200; всего320. Полный ledger и Train ID каждой головы: fit_ledger.json.

| Split | Candidate | Kind | X | Config | Weight | Policy | Biases | Val selected | Val regret | Val F1 | Test accuracy | Test F1 | Test regret | Test success | Test R/MC/PI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| old | old__always_remizov | constant | X | constant | 1 | constant | [0.0, 0.0] |  | 0.0019688878 | 0.2962963 | 90.667% | 0.31701632 | 1.6553834 | 147/150 | 150/0/0 |
| old | old__always_monte_carlo | constant | X | constant | 1 | constant | [0.0, 0.0] |  | 0.018636118 | 0.060606061 | 4.667% | 0.029723992 | 0.013696633 | 141/150 | 0/150/0 |
| old | old__always_product_integral | constant | X | constant | 1 | constant | [0.0, 0.0] |  | 0.033836845 | 0.060606061 | 4.667% | 0.029723992 | 0.050577574 | 110/150 | 0/0/150 |
| old | old__X__classifier__w1__baseline | classifier | X | baseline | 1 | default | [0.0, 0.0] |  | 0.0019688878 | 0.2962963 | 90.667% | 0.31701632 | 1.6553834 | 147/150 | 150/0/0 |
| old | old__X__classifier__w1__baseline | classifier | X | baseline | 1 | validation_optimal | [0.0, 0.8264396519564298] |  | 0.0019278025 | 0.33906444 | 90.667% | 0.31701632 | 1.6553834 | 147/150 | 150/0/0 |
| old | old__X__classifier__w1__medium | classifier | X | medium | 1 | default | [0.0, 0.0] |  | 0.0019688878 | 0.2962963 | 90.667% | 0.31812865 | 1.6553796 | 147/150 | 149/1/0 |
| old | old__X__classifier__w1__medium | classifier | X | medium | 1 | validation_optimal | [0.0, 0.33727778272627956] |  | 0.0019278025 | 0.33906444 | 90.667% | 0.31812865 | 1.6553796 | 147/150 | 149/1/0 |
| old | old__X__classifier__w1__deeper | classifier | X | deeper | 1 | default | [0.0, 0.0] |  | 0.0019688878 | 0.2962963 | 90.667% | 0.31701632 | 1.6553834 | 147/150 | 150/0/0 |
| old | old__X__classifier__w1__deeper | classifier | X | deeper | 1 | validation_optimal | [0.0, 1.169108058185696] |  | 0.0019278025 | 0.33906444 | 89.333% | 0.31455399 | 1.6559604 | 146/150 | 148/0/2 |
| old | old__X__classifier__w2__baseline | classifier | X | baseline | 2 | default | [0.0, 0.0] |  | 0.0021935381 | 0.29491945 | 90.000% | 0.31578947 | 1.6554429 | 147/150 | 149/1/0 |
| old | old__X__classifier__w2__baseline | classifier | X | baseline | 2 | validation_optimal | [-0.09091272366252214, 0.8408439429757538] |  | 0.0019278025 | 0.33906444 | 90.667% | 0.31701632 | 1.6553834 | 147/150 | 150/0/0 |
| old | old__X__classifier__w2__medium | classifier | X | medium | 2 | default | [0.0, 0.0] |  | 0.0021935381 | 0.29491945 | 87.333% | 0.31190476 | 1.6563869 | 146/150 | 144/6/0 |
| old | old__X__classifier__w2__medium | classifier | X | medium | 2 | validation_optimal | [-0.01113557237720675, 0.4054651081081642] |  | 0.0019278025 | 0.33906444 | 87.333% | 0.31190476 | 1.6563571 | 146/150 | 144/5/1 |
| old | old__X__classifier__w2__deeper | classifier | X | deeper | 2 | default | [0.0, 0.0] |  | 0.0021935381 | 0.29491945 | 89.333% | 0.36794872 | 1.6558324 | 147/150 | 144/6/0 |
| old | old__X__classifier__w2__deeper | classifier | X | deeper | 2 | validation_optimal | [-0.6009827211220489, 0.6764443341479108] |  | 0.0019278025 | 0.33906444 | 90.667% | 0.39097548 | 1.6554027 | 147/150 | 148/0/2 |
| old | old__X__classifier__w3__baseline | classifier | X | baseline | 3 | default | [0.0, 0.0] |  | 0.0022615794 | 0.3283208 | 87.333% | 0.31190476 | 1.6564924 | 146/150 | 144/6/0 |
| old | old__X__classifier__w3__baseline | classifier | X | baseline | 3 | validation_optimal | [-0.3739242308465185, 0.0] |  | 0.0019688878 | 0.2962963 | 90.667% | 0.31701632 | 1.6553834 | 147/150 | 150/0/0 |
| old | old__X__classifier__w3__medium | classifier | X | medium | 3 | default | [0.0, 0.0] |  | 0.0021631 | 0.36591479 | 86.667% | 0.31400966 | 1.6563791 | 146/150 | 140/10/0 |
| old | old__X__classifier__w3__medium | classifier | X | medium | 3 | validation_optimal | [-0.39572941437214115, 0.0] |  | 0.0019688878 | 0.2962963 | 88.667% | 0.31330978 | 1.6557398 | 147/150 | 147/3/0 |
| old | old__X__classifier__w3__deeper | classifier | X | deeper | 3 | default | [0.0, 0.0] |  | 0.0023136861 | 0.32515723 | 88.000% | 0.35859313 | 1.6557209 | 147/150 | 142/8/0 |
| old | old__X__classifier__w3__deeper | classifier | X | deeper | 3 | validation_optimal | [-0.2895959997220446, 0.0] |  | 0.0019688878 | 0.2962963 | 89.333% | 0.37502686 | 1.6557301 | 147/150 | 146/4/0 |
| old | old__X__classifier__w5__baseline | classifier | X | baseline | 5 | default | [0.0, 0.0] |  | 0.0040072354 | 0.3486144 | 80.000% | 0.36135057 | 1.6573124 | 145/150 | 125/25/0 |
| old | old__X__classifier__w5__baseline | classifier | X | baseline | 5 | validation_optimal | [-0.7320980751202326, 0.0] |  | 0.0019688878 | 0.2962963 | 90.000% | 0.31578947 | 1.6556238 | 147/150 | 149/1/0 |
| old | old__X__classifier__w5__medium | classifier | X | medium | 5 | default | [0.0, 0.0] |  | 0.002772078 | 0.3479807 | 84.000% | 0.38519375 | 1.6573517 | 145/150 | 130/19/1 |
| old | old__X__classifier__w5__medium | classifier | X | medium | 5 | validation_optimal | [-0.6827228623881476, 0.0] |  | 0.0019688878 | 0.2962963 | 88.667% | 0.31330978 | 1.6558836 | 147/150 | 147/2/1 |
| old | old__X__classifier__w5__deeper | classifier | X | deeper | 5 | default | [0.0, 0.0] |  | 0.003321305 | 0.33400268 | 82.667% | 0.35795256 | 1.6572029 | 145/150 | 131/18/1 |
| old | old__X__classifier__w5__deeper | classifier | X | deeper | 5 | validation_optimal | [-0.5612228949546215, 0.0] |  | 0.0019688878 | 0.2962963 | 89.333% | 0.42539683 | 1.6556804 | 147/150 | 144/5/1 |
| old | old__X__classifier__w8__baseline | classifier | X | baseline | 8 | default | [0.0, 0.0] |  | 0.0070873587 | 0.31735414 | 65.333% | 0.31681877 | 1.659256 | 144/150 | 100/45/5 |
| old | old__X__classifier__w8__baseline | classifier | X | baseline | 8 | validation_optimal | [-0.8953836354727939, -0.029994727304988315] |  | 0.0018692057 | 0.36591479 | 85.333% | 0.35916084 | 1.656574 | 147/150 | 139/5/6 |
| old | old__X__classifier__w8__medium | classifier | X | medium | 8 | default | [0.0, 0.0] |  | 0.0054239319 | 0.31970298 | 72.667% | 0.40420532 | 1.658104 | 145/150 | 109/37/4 |
| old | old__X__classifier__w8__medium | classifier | X | medium | 8 | validation_optimal | [-1.1870981548299846, -0.12342714531342157] |  | 0.0019688878 | 0.2962963 | 89.333% | 0.38220641 | 1.6561429 | 146/150 | 145/2/3 |
| old | old__X__classifier__w8__deeper | classifier | X | deeper | 8 | default | [0.0, 0.0] |  | 0.0051591189 | 0.32265283 | 72.000% | 0.40640552 | 1.658218 | 145/150 | 110/37/3 |
| old | old__X__classifier__w8__deeper | classifier | X | deeper | 8 | validation_optimal | [-1.3386002692640961, -0.1640061673974885] |  | 0.0019688878 | 0.2962963 | 90.667% | 0.46678487 | 1.6554506 | 147/150 | 146/1/3 |
| old | old__X__classifier__w12__baseline | classifier | X | baseline | 12 | default | [0.0, 0.0] |  | 0.014023638 | 0.23988203 | 35.333% | 0.24047764 | 0.013372903 | 136/150 | 47/83/20 |
| old | old__X__classifier__w12__baseline | classifier | X | baseline | 12 | validation_optimal | [-1.4994377492378759, -0.47947970288676794] |  | 0.0018301005 | 0.33906444 | 87.333% | 0.31079478 | 1.6567837 | 145/150 | 145/0/5 |
| old | old__X__classifier__w12__medium | classifier | X | medium | 12 | default | [0.0, 0.0] |  | 0.0081072405 | 0.30065593 | 58.000% | 0.36176176 | 1.6602453 | 144/150 | 86/53/11 |
| old | old__X__classifier__w12__medium | classifier | X | medium | 12 | validation_optimal | [-1.6416154894470065, -0.373123153604022] |  | 0.0019688878 | 0.2962963 | 87.333% | 0.31190476 | 1.6561692 | 146/150 | 144/0/6 |
| old | old__X__classifier__w12__deeper | classifier | X | deeper | 12 | default | [0.0, 0.0] |  | 0.0084521689 | 0.29321042 | 61.333% | 0.35138253 | 1.6613211 | 142/150 | 90/50/10 |
| old | old__X__classifier__w12__deeper | classifier | X | deeper | 12 | validation_optimal | [-1.5242036082602923, -0.09621532227679765] |  | 0.0019688878 | 0.2962963 | 86.667% | 0.42011354 | 1.6574558 | 146/150 | 138/1/11 |
| old | old__X__raw__w1__baseline | raw | X | baseline | 1 | default | [0.0, 0.0] |  | 0.015220525 | 0.07295191 | 5.333% | 0.035977291 | 0.014352105 | 138/150 | 1/143/6 |
| old | old__X__raw__w1__baseline | raw | X | baseline | 1 | validation_optimal | [-0.01549496148677676, -0.0010954577715973637] |  | 0.013585243 | 0.12763992 | 8.000% | 0.085714286 | 0.014543288 | 137/150 | 4/133/13 |
| old | old__X__raw__w1__medium | raw | X | medium | 1 | default | [0.0, 0.0] |  | 0.0073567909 | 0.32148962 | 88.667% | 0.37983393 | 0.0014709113 | 145/150 | 145/3/2 |
| old | old__X__raw__w1__medium | raw | X | medium | 1 | validation_optimal | [-9.107452714842399, -8.967612258968556] |  | 0.0019716892 | 0.29463171 | 90.667% | 0.40023474 | 0.00065461161 | 147/150 | 148/1/1 |
| old | old__X__raw__w1__deeper | raw | X | deeper | 1 | default | [0.0, 0.0] |  | 0.018243574 | 0.17768458 | 16.667% | 0.1099423 | 0.016297258 | 132/150 | 23/106/21 |
| old | old__X__raw__w1__deeper | raw | X | deeper | 1 | validation_optimal | [-0.11363242216123816, -0.18031200599145958] |  | 0.0024570588 | 0.36981342 | 88.667% | 0.37377332 | 0.0012399126 | 146/150 | 145/4/1 |
| old | old__X__asinh__w1__baseline | asinh | X | baseline | 1 | default | [0.0, 0.0] |  | 0.0024629597 | 0.29491945 | 90.667% | 0.31701632 | 1.6553834 | 147/150 | 150/0/0 |
| old | old__X__asinh__w1__baseline | asinh | X | baseline | 1 | validation_optimal | [0.0, 0.0] |  | 0.0024629597 | 0.29491945 | 90.667% | 0.31701632 | 1.6553834 | 147/150 | 150/0/0 |
| old | old__X__asinh__w1__medium | asinh | X | medium | 1 | default | [0.0, 0.0] |  | 0.0027312053 | 0.29353234 | 90.000% | 0.31578947 | 1.6558154 | 146/150 | 149/1/0 |
| old | old__X__asinh__w1__medium | asinh | X | medium | 1 | validation_optimal | [0.0, -0.007631717723766154] |  | 0.0024629597 | 0.29491945 | 90.000% | 0.31578947 | 1.6558154 | 146/150 | 149/1/0 |
| old | old__X__asinh__w1__deeper | asinh | X | deeper | 1 | default | [0.0, 0.0] |  | 0.0027312053 | 0.29353234 | 90.667% | 0.31701632 | 1.6553834 | 147/150 | 150/0/0 |
| old | old__X__asinh__w1__deeper | asinh | X | deeper | 1 | validation_optimal | [0.005177821414521847, -0.0054186143872472945] |  | 0.0024003981 | 0.35059188 | 89.333% | 0.37502686 | 0.00064019608 | 147/150 | 146/4/0 |
| old | old__augmented__classifier__w1__baseline | classifier | augmented | baseline | 1 | default | [0.0, 0.0] |  | 0.0019448626 | 0.29739777 | 90.667% | 0.31701632 | 1.6553834 | 147/150 | 150/0/0 |
| old | old__augmented__classifier__w1__baseline | classifier | augmented | baseline | 1 | validation_optimal | [0.0, 0.7115150650094448] |  | 0.0019278025 | 0.33906444 | 90.667% | 0.31701632 | 1.6553834 | 147/150 | 150/0/0 |
| old | old__augmented__classifier__w1__medium | classifier | augmented | medium | 1 | default | [0.0, 0.0] |  | 0.0019688878 | 0.2962963 | 90.667% | 0.31812865 | 1.6553796 | 147/150 | 149/1/0 |
| old | old__augmented__classifier__w1__medium | classifier | augmented | medium | 1 | validation_optimal | [0.0, 0.4054651081081642] |  | 0.0019278025 | 0.33906444 | 90.667% | 0.31812865 | 1.6553796 | 147/150 | 149/1/0 |
| old | old__augmented__classifier__w1__deeper | classifier | augmented | deeper | 1 | default | [0.0, 0.0] |  | 0.0019688878 | 0.2962963 | 90.000% | 0.31690141 | 1.6554392 | 147/150 | 148/2/0 |
| old | old__augmented__classifier__w1__deeper | classifier | augmented | deeper | 1 | validation_optimal | [0.0, 0.8158710381418046] |  | 0.0019278025 | 0.33906444 | 90.000% | 0.31690141 | 1.6554392 | 147/150 | 148/2/0 |
| old | old__augmented__classifier__w2__baseline | classifier | augmented | baseline | 2 | default | [0.0, 0.0] |  | 0.0019688878 | 0.2962963 | 90.000% | 0.31578947 | 1.6554429 | 147/150 | 149/1/0 |
| old | old__augmented__classifier__w2__baseline | classifier | augmented | baseline | 2 | validation_optimal | [0.6053029243935057, 0.0] |  | 0.0019134573 | 0.3774359 | 85.333% | 0.37056916 | 1.656358 | 146/150 | 135/15/0 |
| old | old__augmented__classifier__w2__medium | classifier | augmented | medium | 2 | default | [0.0, 0.0] |  | 0.0019688878 | 0.2962963 | 88.667% | 0.3144208 | 1.6556829 | 147/150 | 146/4/0 |
| old | old__augmented__classifier__w2__medium | classifier | augmented | medium | 2 | validation_optimal | [0.0, 0.26852438652064653] |  | 0.0019278025 | 0.33906444 | 88.667% | 0.3144208 | 1.6556829 | 147/150 | 146/4/0 |
| old | old__augmented__classifier__w2__deeper | classifier | augmented | deeper | 2 | default | [0.0, 0.0] |  | 0.0021935381 | 0.29491945 | 90.000% | 0.37739093 | 1.6555297 | 147/150 | 146/4/0 |
| old | old__augmented__classifier__w2__deeper | classifier | augmented | deeper | 2 | validation_optimal | [-0.4911313089248091, 0.5635793958077069] |  | 0.0019278025 | 0.33906444 | 89.333% | 0.31455399 | 1.6554696 | 147/150 | 148/1/1 |
| old | old__augmented__classifier__w3__baseline | classifier | augmented | baseline | 3 | default | [0.0, 0.0] |  | 0.0022823345 | 0.29463171 | 88.000% | 0.31428571 | 1.6556167 | 147/150 | 144/6/0 |
| old | old__augmented__classifier__w3__baseline | classifier | augmented | baseline | 3 | validation_optimal | [0.2848367613789384, 0.0] | ДА | 0.0017860649 | 0.36864772 | 84.000% | 0.33780611 | 1.6567536 | 146/150 | 135/15/0 |
| old | old__augmented__classifier__w3__medium | classifier | augmented | medium | 3 | default | [0.0, 0.0] |  | 0.002348273 | 0.36591479 | 86.667% | 0.31287605 | 1.6556861 | 147/150 | 141/9/0 |
| old | old__augmented__classifier__w3__medium | classifier | augmented | medium | 3 | validation_optimal | [-0.23765601201174869, 0.3143370076954712] |  | 0.0017890152 | 0.3818408 | 86.667% | 0.31063321 | 1.655695 | 147/150 | 143/6/1 |
| old | old__augmented__classifier__w3__deeper | classifier | augmented | deeper | 3 | default | [0.0, 0.0] |  | 0.0020425583 | 0.37120266 | 88.667% | 0.39861613 | 1.6556401 | 147/150 | 141/9/0 |
| old | old__augmented__classifier__w3__deeper | classifier | augmented | deeper | 3 | validation_optimal | [-0.24921119085323043, 0.633778745190333] |  | 0.0017890152 | 0.3818408 | 88.000% | 0.36430475 | 1.655682 | 147/150 | 143/6/1 |
| old | old__augmented__classifier__w5__baseline | classifier | augmented | baseline | 5 | default | [0.0, 0.0] |  | 0.002638424 | 0.35920047 | 78.667% | 0.35483383 | 1.657336 | 145/150 | 123/27/0 |
| old | old__augmented__classifier__w5__baseline | classifier | augmented | baseline | 5 | validation_optimal | [-0.6310630583952475, 0.0] |  | 0.0019688878 | 0.2962963 | 89.333% | 0.31455399 | 1.655724 | 147/150 | 148/2/0 |
| old | old__augmented__classifier__w5__medium | classifier | augmented | medium | 5 | default | [0.0, 0.0] |  | 0.0033113497 | 0.36965998 | 84.667% | 0.43924261 | 1.6563733 | 147/150 | 131/17/2 |
| old | old__augmented__classifier__w5__medium | classifier | augmented | medium | 5 | validation_optimal | [-0.8783105508353948, 0.0] |  | 0.0019688878 | 0.2962963 | 89.333% | 0.38849488 | 1.6558793 | 147/150 | 146/2/2 |
| old | old__augmented__classifier__w5__deeper | classifier | augmented | deeper | 5 | default | [0.0, 0.0] |  | 0.0030085913 | 0.34444993 | 84.667% | 0.46234475 | 1.6569932 | 146/150 | 130/18/2 |
| old | old__augmented__classifier__w5__deeper | classifier | augmented | deeper | 5 | validation_optimal | [-0.4847974498488011, 0.0] |  | 0.0018301005 | 0.33906444 | 90.000% | 0.50059737 | 1.6557292 | 147/150 | 143/5/2 |
| old | old__augmented__classifier__w8__baseline | classifier | augmented | baseline | 8 | default | [0.0, 0.0] |  | 0.007046928 | 0.33751178 | 65.333% | 0.31681877 | 1.6596741 | 143/150 | 100/45/5 |
| old | old__augmented__classifier__w8__baseline | classifier | augmented | baseline | 8 | validation_optimal | [-0.807992187133059, -0.01317601032350524] |  | 0.0019389649 | 0.37120266 | 84.667% | 0.35785139 | 1.6564543 | 147/150 | 138/6/6 |
| old | old__augmented__classifier__w8__medium | classifier | augmented | medium | 8 | default | [0.0, 0.0] |  | 0.00412274 | 0.31679238 | 72.667% | 0.40891912 | 1.6577911 | 146/150 | 109/38/3 |
| old | old__augmented__classifier__w8__medium | classifier | augmented | medium | 8 | validation_optimal | [-1.1429750820473448, -0.03882641788860864] |  | 0.0019688878 | 0.2962963 | 89.333% | 0.46912895 | 1.6559606 | 147/150 | 141/3/6 |
| old | old__augmented__classifier__w8__deeper | classifier | augmented | deeper | 8 | default | [0.0, 0.0] |  | 0.0050279426 | 0.32148148 | 74.000% | 0.45915602 | 1.6589287 | 143/150 | 110/35/5 |
| old | old__augmented__classifier__w8__deeper | classifier | augmented | deeper | 8 | validation_optimal | [-1.2587048348458025, 0.0] |  | 0.0019688878 | 0.2962963 | 91.333% | 0.55498116 | 1.655704 | 147/150 | 143/1/6 |
| old | old__augmented__classifier__w12__baseline | classifier | augmented | baseline | 12 | default | [0.0, 0.0] |  | 0.015599702 | 0.24365942 | 35.333% | 0.26096371 | 0.013431463 | 137/150 | 45/84/21 |
| old | old__augmented__classifier__w12__baseline | classifier | augmented | baseline | 12 | validation_optimal | [-1.496895841877645, -0.4949876861899432] |  | 0.0018301005 | 0.33906444 | 88.000% | 0.31205674 | 1.6564142 | 146/150 | 146/1/3 |
| old | old__augmented__classifier__w12__medium | classifier | augmented | medium | 12 | default | [0.0, 0.0] |  | 0.01019709 | 0.31319286 | 56.000% | 0.32093767 | 1.6613589 | 142/150 | 84/56/10 |
| old | old__augmented__classifier__w12__medium | classifier | augmented | medium | 12 | validation_optimal | [-1.700859598682607, -0.385675744147979] |  | 0.0019688878 | 0.2962963 | 88.000% | 0.31316726 | 1.6561662 | 146/150 | 145/0/5 |
| old | old__augmented__classifier__w12__deeper | classifier | augmented | deeper | 12 | default | [0.0, 0.0] |  | 0.0081023164 | 0.28944345 | 61.333% | 0.35282525 | 0.007023928 | 141/150 | 90/51/9 |
| old | old__augmented__classifier__w12__deeper | classifier | augmented | deeper | 12 | validation_optimal | [-1.5747299089870823, -0.28925524604109165] |  | 0.0019688878 | 0.2962963 | 88.667% | 0.37377332 | 1.6566308 | 147/150 | 145/1/4 |
| old | old__augmented__raw__w1__baseline | raw | augmented | baseline | 1 | default | [0.0, 0.0] |  | 0.015220525 | 0.07295191 | 5.333% | 0.035977291 | 0.014352105 | 138/150 | 1/143/6 |
| old | old__augmented__raw__w1__baseline | raw | augmented | baseline | 1 | validation_optimal | [-0.015472014985399296, -0.0013623069031467596] |  | 0.013594058 | 0.12946914 | 6.667% | 0.047022522 | 0.014720004 | 137/150 | 3/136/11 |
| old | old__augmented__raw__w1__medium | raw | augmented | medium | 1 | default | [0.0, 0.0] |  | 0.0073567909 | 0.32148962 | 88.667% | 0.37983393 | 0.0014709113 | 145/150 | 145/3/2 |
| old | old__augmented__raw__w1__medium | raw | augmented | medium | 1 | validation_optimal | [-9.107452714842399, -8.967612258968556] |  | 0.0019716892 | 0.29463171 | 90.667% | 0.40023474 | 0.00065461161 | 147/150 | 148/1/1 |
| old | old__augmented__raw__w1__deeper | raw | augmented | deeper | 1 | default | [0.0, 0.0] |  | 0.01929126 | 0.17739522 | 16.000% | 0.1059989 | 0.016666924 | 131/150 | 22/108/20 |
| old | old__augmented__raw__w1__deeper | raw | augmented | deeper | 1 | validation_optimal | [-0.11471000043307457, -0.18031200599145958] |  | 0.002459502 | 0.3338474 | 88.667% | 0.37377332 | 0.0012399126 | 146/150 | 145/4/1 |
| old | old__augmented__asinh__w1__baseline | asinh | augmented | baseline | 1 | default | [0.0, 0.0] |  | 0.0024629597 | 0.29491945 | 90.667% | 0.31701632 | 1.6553834 | 147/150 | 150/0/0 |
| old | old__augmented__asinh__w1__baseline | asinh | augmented | baseline | 1 | validation_optimal | [0.0, 0.0] |  | 0.0024629597 | 0.29491945 | 90.667% | 0.31701632 | 1.6553834 | 147/150 | 150/0/0 |
| old | old__augmented__asinh__w1__medium | asinh | augmented | medium | 1 | default | [0.0, 0.0] |  | 0.0027312053 | 0.29353234 | 90.000% | 0.31578947 | 1.6558154 | 146/150 | 149/1/0 |
| old | old__augmented__asinh__w1__medium | asinh | augmented | medium | 1 | validation_optimal | [0.0, -0.0070938560518252645] |  | 0.0024629597 | 0.29491945 | 90.000% | 0.31578947 | 1.6558154 | 146/150 | 149/1/0 |
| old | old__augmented__asinh__w1__deeper | asinh | augmented | deeper | 1 | default | [0.0, 0.0] |  | 0.0027312053 | 0.29353234 | 90.667% | 0.31701632 | 1.6553834 | 147/150 | 150/0/0 |
| old | old__augmented__asinh__w1__deeper | asinh | augmented | deeper | 1 | validation_optimal | [0.004929239552286398, -0.005670155524786344] |  | 0.0022778995 | 0.36591479 | 91.333% | 0.40146199 | 0.0005833943 | 147/150 | 149/1/0 |
| new | new__always_remizov | constant | X | constant | 1 | constant | [0.0, 0.0] |  | 0.0023343358 | 0.3003663 | 90.667% | 0.31701632 | 1.6553834 | 147/150 | 150/0/0 |
| new | new__always_monte_carlo | constant | X | constant | 1 | constant | [0.0, 0.0] |  | 0.011044761 | 0.071428571 | 4.667% | 0.029723992 | 0.013696633 | 141/150 | 0/150/0 |
| new | new__always_product_integral | constant | X | constant | 1 | constant | [0.0, 0.0] |  | 0.037001425 | 0.037735849 | 4.667% | 0.029723992 | 0.050577574 | 110/150 | 0/0/150 |
| new | new__X__classifier__w1__baseline | classifier | X | baseline | 1 | default | [0.0, 0.0] |  | 0.0023343358 | 0.3003663 | 90.667% | 0.31701632 | 1.6553834 | 147/150 | 150/0/0 |
| new | new__X__classifier__w1__baseline | classifier | X | baseline | 1 | validation_optimal | [0.0, 1.9013915755559234] |  | 0.00081818526 | 0.39726388 | 90.000% | 0.3180212 | 1.6571723 | 145/150 | 147/0/3 |
| new | new__X__classifier__w1__medium | classifier | X | medium | 1 | default | [0.0, 0.0] |  | 0.0023343358 | 0.3003663 | 90.667% | 0.31701632 | 1.6553834 | 147/150 | 150/0/0 |
| new | new__X__classifier__w1__medium | classifier | X | medium | 1 | validation_optimal | [0.0, 1.4406255473682337] |  | 0.0011696905 | 0.37575005 | 90.667% | 0.31701632 | 1.6553834 | 147/150 | 150/0/0 |
| new | new__X__classifier__w1__deeper | classifier | X | deeper | 1 | default | [0.0, 0.0] |  | 0.0023343358 | 0.3003663 | 90.667% | 0.31701632 | 1.6553834 | 147/150 | 150/0/0 |
| new | new__X__classifier__w1__deeper | classifier | X | deeper | 1 | validation_optimal | [1.2917202952091427, 0.0] |  | 0.0013115144 | 0.40268199 | 84.000% | 0.36564911 | 1.6557715 | 147/150 | 135/15/0 |
| new | new__X__classifier__w2__baseline | classifier | X | baseline | 2 | default | [0.0, 0.0] |  | 0.0023343358 | 0.3003663 | 90.667% | 0.31701632 | 1.6553834 | 147/150 | 150/0/0 |
| new | new__X__classifier__w2__baseline | classifier | X | baseline | 2 | validation_optimal | [0.0, 0.0] |  | 0.0023343358 | 0.3003663 | 90.667% | 0.31701632 | 1.6553834 | 147/150 | 150/0/0 |
| new | new__X__classifier__w2__medium | classifier | X | medium | 2 | default | [0.0, 0.0] |  | 0.0025459496 | 0.34761905 | 90.000% | 0.31578947 | 1.6553867 | 147/150 | 149/1/0 |
| new | new__X__classifier__w2__medium | classifier | X | medium | 2 | validation_optimal | [-0.19782528429510712, 0.0] |  | 0.0023343358 | 0.3003663 | 90.667% | 0.31701632 | 1.6553834 | 147/150 | 150/0/0 |
| new | new__X__classifier__w2__deeper | classifier | X | deeper | 2 | default | [0.0, 0.0] |  | 0.0023647806 | 0.30202578 | 89.333% | 0.31455399 | 1.6554574 | 147/150 | 148/2/0 |
| new | new__X__classifier__w2__deeper | classifier | X | deeper | 2 | validation_optimal | [0.3119569280292576, 0.0] |  | 0.0022180079 | 0.35132275 | 86.000% | 0.37917787 | 1.6559765 | 147/150 | 138/12/0 |
| new | new__X__classifier__w3__baseline | classifier | X | baseline | 3 | default | [0.0, 0.0] |  | 0.0025461848 | 0.34239603 | 89.333% | 0.37502686 | 1.6559946 | 147/150 | 146/4/0 |
| new | new__X__classifier__w3__baseline | classifier | X | baseline | 3 | validation_optimal | [-0.6047995343877327, 0.0] |  | 0.0023343358 | 0.3003663 | 90.667% | 0.31701632 | 1.6553834 | 147/150 | 150/0/0 |
| new | new__X__classifier__w3__medium | classifier | X | medium | 3 | default | [0.0, 0.0] |  | 0.002593339 | 0.3846516 | 86.667% | 0.35491376 | 1.6556442 | 147/150 | 141/8/1 |
| new | new__X__classifier__w3__medium | classifier | X | medium | 3 | validation_optimal | [-1.0257241228332845, -0.32213304909024465] |  | 0.0023343358 | 0.3003663 | 90.667% | 0.31701632 | 1.6553834 | 147/150 | 150/0/0 |
| new | new__X__classifier__w3__deeper | classifier | X | deeper | 3 | default | [0.0, 0.0] |  | 0.0026834803 | 0.34239603 | 86.000% | 0.30935252 | 1.6556633 | 147/150 | 142/8/0 |
| new | new__X__classifier__w3__deeper | classifier | X | deeper | 3 | validation_optimal | [-0.445887167265782, 0.0] |  | 0.0023343358 | 0.3003663 | 90.000% | 0.31578947 | 1.655454 | 147/150 | 149/1/0 |
| new | new__X__classifier__w5__baseline | classifier | X | baseline | 5 | default | [0.0, 0.0] |  | 0.0038659651 | 0.36005014 | 76.667% | 0.34608733 | 1.6571723 | 146/150 | 121/29/0 |
| new | new__X__classifier__w5__baseline | classifier | X | baseline | 5 | validation_optimal | [-1.0282285665653277, 0.3123546613116119] |  | 0.0023202565 | 0.39726388 | 89.333% | 0.31455399 | 1.6554142 | 147/150 | 148/0/2 |
| new | new__X__classifier__w5__medium | classifier | X | medium | 5 | default | [0.0, 0.0] |  | 0.0097872068 | 0.38159371 | 78.000% | 0.31785262 | 1.6570895 | 146/150 | 127/21/2 |
| new | new__X__classifier__w5__medium | classifier | X | medium | 5 | validation_optimal | [0.19954715901020845, -0.16452821415118454] |  | 0.0046690785 | 0.35760601 | 71.333% | 0.3122178 | 1.6579773 | 145/150 | 115/33/2 |
| new | new__X__classifier__w5__deeper | classifier | X | deeper | 5 | default | [0.0, 0.0] |  | 0.003878711 | 0.36343313 | 82.000% | 0.4229741 | 1.656626 | 146/150 | 130/17/3 |
| new | new__X__classifier__w5__deeper | classifier | X | deeper | 5 | validation_optimal | [-0.9196545119097204, -0.10449856425642677] |  | 0.0023343358 | 0.3003663 | 90.000% | 0.38233216 | 1.6554189 | 147/150 | 147/0/3 |
| new | new__X__classifier__w8__baseline | classifier | X | baseline | 8 | default | [0.0, 0.0] |  | 0.0067787263 | 0.2962963 | 60.667% | 0.31534392 | 0.0072226084 | 142/150 | 89/56/5 |
| new | new__X__classifier__w8__baseline | classifier | X | baseline | 8 | validation_optimal | [-1.3414237380345408, -0.32619185360618747] |  | 0.0023343358 | 0.3003663 | 90.667% | 0.31701632 | 1.6553834 | 147/150 | 150/0/0 |
| new | new__X__classifier__w8__medium | classifier | X | medium | 8 | default | [0.0, 0.0] |  | 0.012100103 | 0.37706856 | 68.667% | 0.42869455 | 1.6583231 | 145/150 | 102/42/6 |
| new | new__X__classifier__w8__medium | classifier | X | medium | 8 | validation_optimal | [0.7679155717774887, 0.0] |  | 0.0072359882 | 0.3557599 | 44.000% | 0.30854205 | 0.0095666229 | 141/150 | 59/87/4 |
| new | new__X__classifier__w8__deeper | classifier | X | deeper | 8 | default | [0.0, 0.0] |  | 0.01191151 | 0.37449091 | 70.667% | 0.36235181 | 1.6578833 | 146/150 | 112/32/6 |
| new | new__X__classifier__w8__deeper | classifier | X | deeper | 8 | validation_optimal | [0.5883501596053393, -0.2119581504043764] |  | 0.0058194935 | 0.40062035 | 57.333% | 0.30604938 | 0.0073691776 | 144/150 | 80/68/2 |
| new | new__X__classifier__w12__baseline | classifier | X | baseline | 12 | default | [0.0, 0.0] |  | 0.0088938807 | 0.24205294 | 33.333% | 0.2466815 | 0.014197236 | 138/150 | 41/83/26 |
| new | new__X__classifier__w12__baseline | classifier | X | baseline | 12 | validation_optimal | [-1.5389305611462787, -0.9284780102848469] |  | 0.0023343358 | 0.3003663 | 90.667% | 0.31701632 | 1.6553834 | 147/150 | 150/0/0 |
| new | new__X__classifier__w12__medium | classifier | X | medium | 12 | default | [0.0, 0.0] |  | 0.014548421 | 0.32215234 | 51.333% | 0.33599938 | 0.011318298 | 141/150 | 70/63/17 |
| new | new__X__classifier__w12__medium | classifier | X | medium | 12 | validation_optimal | [-1.38979459648237, -0.9697452516997777] |  | 0.0023343358 | 0.3003663 | 89.333% | 0.31566549 | 1.6552816 | 147/150 | 147/2/1 |
| new | new__X__classifier__w12__deeper | classifier | X | deeper | 12 | default | [0.0, 0.0] |  | 0.016872131 | 0.29936121 | 56.000% | 0.35986619 | 0.0085037819 | 141/150 | 78/57/15 |
| new | new__X__classifier__w12__deeper | classifier | X | deeper | 12 | validation_optimal | [-1.4160548721407555, -0.730493750926019] |  | 0.0023274266 | 0.39726388 | 89.333% | 0.31566549 | 1.6552481 | 147/150 | 147/1/2 |
| new | new__X__raw__w1__baseline | raw | X | baseline | 1 | default | [0.0, 0.0] |  | 0.010809565 | 0.080104201 | 5.333% | 0.035736117 | 1.6699318 | 142/150 | 3/146/1 |
| new | new__X__raw__w1__baseline | raw | X | baseline | 1 | validation_optimal | [-0.014384247710361286, 0.0] |  | 0.010105922 | 0.12952175 | 12.667% | 0.087189542 | 1.6702549 | 139/150 | 14/129/7 |
| new | new__X__raw__w1__medium | raw | X | medium | 1 | default | [0.0, 0.0] |  | 0.002432999 | 0.2962963 | 88.667% | 0.31330978 | 1.6578295 | 146/150 | 147/3/0 |
| new | new__X__raw__w1__medium | raw | X | medium | 1 | validation_optimal | [-0.0021722854169956562, 0.0] |  | 0.0023343358 | 0.3003663 | 88.667% | 0.31330978 | 1.6578295 | 146/150 | 147/3/0 |
| new | new__X__raw__w1__deeper | raw | X | deeper | 1 | default | [0.0, 0.0] |  | 0.0066639363 | 0.27392977 | 62.667% | 0.25895317 | 1.6652689 | 142/150 | 106/33/11 |
| new | new__X__raw__w1__deeper | raw | X | deeper | 1 | validation_optimal | [-0.093458476254489, -0.06506708803419926] |  | 0.0023343358 | 0.3003663 | 88.000% | 0.31205674 | 1.6599455 | 146/150 | 146/3/1 |
| new | new__X__asinh__w1__baseline | asinh | X | baseline | 1 | default | [0.0, 0.0] |  | 0.0028324343 | 0.2962963 | 90.000% | 0.31690141 | 0.0024899792 | 146/150 | 148/1/1 |
| new | new__X__asinh__w1__baseline | asinh | X | baseline | 1 | validation_optimal | [-0.015495102484111147, 0.0] |  | 0.0023343358 | 0.3003663 | 90.667% | 0.31812865 | 0.0020579793 | 147/150 | 149/0/1 |
| new | new__X__asinh__w1__medium | asinh | X | medium | 1 | default | [0.0, 0.0] |  | 0.0028324343 | 0.2962963 | 89.333% | 0.31566549 | 0.002870213 | 145/150 | 147/2/1 |
| new | new__X__asinh__w1__medium | asinh | X | medium | 1 | validation_optimal | [-0.005462153741604688, 0.0] |  | 0.0023343358 | 0.3003663 | 89.333% | 0.31566549 | 0.002870213 | 145/150 | 147/2/1 |
| new | new__X__asinh__w1__deeper | asinh | X | deeper | 1 | default | [0.0, 0.0] |  | 0.0028324343 | 0.2962963 | 90.000% | 0.31690141 | 0.0024899792 | 146/150 | 148/1/1 |
| new | new__X__asinh__w1__deeper | asinh | X | deeper | 1 | validation_optimal | [-0.002884927386207427, 0.0] |  | 0.0023343358 | 0.3003663 | 90.000% | 0.31690141 | 0.0024899792 | 146/150 | 148/1/1 |
| new | new__augmented__classifier__w1__baseline | classifier | augmented | baseline | 1 | default | [0.0, 0.0] |  | 0.0023343358 | 0.3003663 | 90.667% | 0.31701632 | 1.6553834 | 147/150 | 150/0/0 |
| new | new__augmented__classifier__w1__baseline | classifier | augmented | baseline | 1 | validation_optimal | [1.1865581967736538, 1.844231170778237] |  | 0.0011165356 | 0.46599327 | 88.667% | 0.4082608 | 1.6556671 | 147/150 | 143/7/0 |
| new | new__augmented__classifier__w1__medium | classifier | augmented | medium | 1 | default | [0.0, 0.0] |  | 0.0023343358 | 0.3003663 | 90.667% | 0.31701632 | 1.6553834 | 147/150 | 150/0/0 |
| new | new__augmented__classifier__w1__medium | classifier | augmented | medium | 1 | validation_optimal | [0.7585894996714374, 1.4368175228112068] | ДА | 0.00075184309 | 0.43432317 | 88.667% | 0.3144208 | 1.6552717 | 147/150 | 146/4/0 |
| new | new__augmented__classifier__w1__deeper | classifier | augmented | deeper | 1 | default | [0.0, 0.0] |  | 0.0023343358 | 0.3003663 | 90.667% | 0.31701632 | 1.6553834 | 147/150 | 150/0/0 |
| new | new__augmented__classifier__w1__deeper | classifier | augmented | deeper | 1 | validation_optimal | [1.387214050747918, 0.0] |  | 0.0016267675 | 0.46374753 | 82.667% | 0.35681592 | 1.6558421 | 147/150 | 132/18/0 |
| new | new__augmented__classifier__w2__baseline | classifier | augmented | baseline | 2 | default | [0.0, 0.0] |  | 0.0023343358 | 0.3003663 | 90.667% | 0.31701632 | 1.6553834 | 147/150 | 150/0/0 |
| new | new__augmented__classifier__w2__baseline | classifier | augmented | baseline | 2 | validation_optimal | [0.7116697176895076, 1.4299742573717271] |  | 0.0016632887 | 0.54206349 | 80.667% | 0.36173768 | 1.6579858 | 145/150 | 130/14/6 |
| new | new__augmented__classifier__w2__medium | classifier | augmented | medium | 2 | default | [0.0, 0.0] |  | 0.0025459496 | 0.34761905 | 89.333% | 0.37502686 | 1.6555479 | 147/150 | 146/4/0 |
| new | new__augmented__classifier__w2__medium | classifier | augmented | medium | 2 | validation_optimal | [-0.3390858374700274, 0.0] |  | 0.0023343358 | 0.3003663 | 90.667% | 0.31701632 | 1.6553834 | 147/150 | 150/0/0 |
| new | new__augmented__classifier__w2__deeper | classifier | augmented | deeper | 2 | default | [0.0, 0.0] |  | 0.0023647806 | 0.30202578 | 88.667% | 0.36556777 | 1.6553327 | 147/150 | 144/6/0 |
| new | new__augmented__classifier__w2__deeper | classifier | augmented | deeper | 2 | validation_optimal | [0.7936089105855292, 0.0] |  | 0.0016649954 | 0.40915033 | 80.667% | 0.34648206 | 1.6569099 | 146/150 | 128/22/0 |
| new | new__augmented__classifier__w3__baseline | classifier | augmented | baseline | 3 | default | [0.0, 0.0] |  | 0.0025461848 | 0.34239603 | 90.000% | 0.38233216 | 1.6557144 | 147/150 | 147/3/0 |
| new | new__augmented__classifier__w3__baseline | classifier | augmented | baseline | 3 | validation_optimal | [-0.6015725561701488, 0.0] |  | 0.0023343358 | 0.3003663 | 90.667% | 0.31701632 | 1.6553834 | 147/150 | 150/0/0 |
| new | new__augmented__classifier__w3__medium | classifier | augmented | medium | 3 | default | [0.0, 0.0] |  | 0.002593339 | 0.3846516 | 87.333% | 0.35936965 | 1.6553612 | 147/150 | 142/7/1 |
| new | new__augmented__classifier__w3__medium | classifier | augmented | medium | 3 | validation_optimal | [-0.8697140379071032, -0.37530855714974787] |  | 0.0023343358 | 0.3003663 | 90.667% | 0.31701632 | 1.6553834 | 147/150 | 150/0/0 |
| new | new__augmented__classifier__w3__deeper | classifier | augmented | deeper | 3 | default | [0.0, 0.0] |  | 0.0026834803 | 0.34239603 | 84.667% | 0.34249158 | 1.6559413 | 147/150 | 139/11/0 |
| new | new__augmented__classifier__w3__deeper | classifier | augmented | deeper | 3 | validation_optimal | [-0.34436031256360156, 0.0] |  | 0.0023343358 | 0.3003663 | 88.667% | 0.31330978 | 1.6555289 | 147/150 | 147/3/0 |
| new | new__augmented__classifier__w5__baseline | classifier | augmented | baseline | 5 | default | [0.0, 0.0] |  | 0.0039140324 | 0.35410001 | 77.333% | 0.34913252 | 1.6571016 | 146/150 | 122/28/0 |
| new | new__augmented__classifier__w5__baseline | classifier | augmented | baseline | 5 | validation_optimal | [-1.0184968400977235, 0.22863814533448154] |  | 0.0023202565 | 0.39726388 | 90.000% | 0.31578947 | 1.6554101 | 147/150 | 149/0/1 |
| new | new__augmented__classifier__w5__medium | classifier | augmented | medium | 5 | default | [0.0, 0.0] |  | 0.003812238 | 0.37109748 | 79.333% | 0.34419699 | 1.657002 | 146/150 | 127/21/2 |
| new | new__augmented__classifier__w5__medium | classifier | augmented | medium | 5 | validation_optimal | [-0.9653088487927548, -0.3759974639761746] |  | 0.0023343358 | 0.3003663 | 89.333% | 0.31455399 | 1.6554232 | 147/150 | 148/0/2 |
| new | new__augmented__classifier__w5__deeper | classifier | augmented | deeper | 5 | default | [0.0, 0.0] |  | 0.0035333296 | 0.38373712 | 80.667% | 0.41592852 | 1.6572283 | 145/150 | 128/19/3 |
| new | new__augmented__classifier__w5__deeper | classifier | augmented | deeper | 5 | validation_optimal | [-0.8409983325143131, -0.25389578702592913] |  | 0.0023343358 | 0.3003663 | 90.667% | 0.31701632 | 1.6553834 | 147/150 | 150/0/0 |
| new | new__augmented__classifier__w8__baseline | classifier | augmented | baseline | 8 | default | [0.0, 0.0] |  | 0.0067144341 | 0.31347289 | 62.000% | 0.32217327 | 0.0071325164 | 142/150 | 91/53/6 |
| new | new__augmented__classifier__w8__baseline | classifier | augmented | baseline | 8 | validation_optimal | [-1.37348867893904, -0.3912312927915952] |  | 0.0023343358 | 0.3003663 | 90.667% | 0.31701632 | 1.6553834 | 147/150 | 150/0/0 |
| new | new__augmented__classifier__w8__medium | classifier | augmented | medium | 8 | default | [0.0, 0.0] |  | 0.012029828 | 0.36276111 | 70.000% | 0.43924741 | 1.6578008 | 146/150 | 106/40/4 |
| new | new__augmented__classifier__w8__medium | classifier | augmented | medium | 8 | validation_optimal | [-1.2529178002820573, -0.42789340748046506] |  | 0.0023343358 | 0.3003663 | 89.333% | 0.31455399 | 1.6554232 | 147/150 | 148/0/2 |
| new | new__augmented__classifier__w8__deeper | classifier | augmented | deeper | 8 | default | [0.0, 0.0] |  | 0.0058572598 | 0.38858364 | 69.333% | 0.37108502 | 1.6584151 | 145/150 | 107/37/6 |
| new | new__augmented__classifier__w8__deeper | classifier | augmented | deeper | 8 | validation_optimal | [-1.275093537927446, -0.49657433077423724] |  | 0.0023274266 | 0.39726388 | 89.333% | 0.31455399 | 1.6554232 | 147/150 | 148/0/2 |
| new | new__augmented__classifier__w12__baseline | classifier | augmented | baseline | 12 | default | [0.0, 0.0] |  | 0.0091910287 | 0.23566939 | 32.667% | 0.24390646 | 0.013697447 | 139/150 | 40/85/25 |
| new | new__augmented__classifier__w12__baseline | classifier | augmented | baseline | 12 | validation_optimal | [-1.4863991856847807, -0.901528818305914] |  | 0.0023343358 | 0.3003663 | 90.667% | 0.31701632 | 1.6553834 | 147/150 | 150/0/0 |
| new | new__augmented__classifier__w12__medium | classifier | augmented | medium | 12 | default | [0.0, 0.0] |  | 0.014166084 | 0.28087897 | 48.667% | 0.3419016 | 0.011799999 | 140/150 | 65/65/20 |
| new | new__augmented__classifier__w12__medium | classifier | augmented | medium | 12 | validation_optimal | [-1.4618027285000381, -1.0097214827916028] |  | 0.0023343358 | 0.3003663 | 88.667% | 0.31330978 | 1.6554698 | 147/150 | 147/1/2 |
| new | new__augmented__classifier__w12__deeper | classifier | augmented | deeper | 12 | default | [0.0, 0.0] |  | 0.016288182 | 0.31908832 | 57.333% | 0.36990741 | 0.0085222402 | 141/150 | 80/57/13 |
| new | new__augmented__classifier__w12__deeper | classifier | augmented | deeper | 12 | validation_optimal | [-1.3351398763292572, -0.8482802643632672] |  | 0.0023343358 | 0.3003663 | 88.667% | 0.31330978 | 1.6554698 | 147/150 | 147/1/2 |
| new | new__augmented__raw__w1__baseline | raw | augmented | baseline | 1 | default | [0.0, 0.0] |  | 0.011044761 | 0.071428571 | 6.000% | 0.040601504 | 1.6696276 | 142/150 | 4/145/1 |
| new | new__augmented__raw__w1__baseline | raw | augmented | baseline | 1 | validation_optimal | [-0.01424954684658597, 0.0] |  | 0.0098443978 | 0.21361656 | 10.667% | 0.074336981 | 1.6703858 | 139/150 | 11/131/8 |
| new | new__augmented__raw__w1__medium | raw | augmented | medium | 1 | default | [0.0, 0.0] |  | 0.0023662054 | 0.29834254 | 88.000% | 0.31205674 | 1.6580856 | 146/150 | 146/4/0 |
| new | new__augmented__raw__w1__medium | raw | augmented | medium | 1 | validation_optimal | [-0.0011932947919284231, 0.0] |  | 0.0023343358 | 0.3003663 | 88.667% | 0.31330978 | 1.6578295 | 146/150 | 147/3/0 |
| new | new__augmented__raw__w1__deeper | raw | augmented | deeper | 1 | default | [0.0, 0.0] |  | 0.0072473558 | 0.27124183 | 63.333% | 0.260631 | 1.6652597 | 142/150 | 107/33/10 |
| new | new__augmented__raw__w1__deeper | raw | augmented | deeper | 1 | validation_optimal | [-0.09201643950247806, -0.03566964452598965] |  | 0.0023343358 | 0.3003663 | 86.667% | 0.31063321 | 1.6615763 | 145/150 | 143/4/3 |
| new | new__augmented__asinh__w1__baseline | asinh | augmented | baseline | 1 | default | [0.0, 0.0] |  | 0.0028324343 | 0.2962963 | 90.000% | 0.31690141 | 0.0024899792 | 146/150 | 148/1/1 |
| new | new__augmented__asinh__w1__baseline | asinh | augmented | baseline | 1 | validation_optimal | [-0.016699867145507686, 0.0] |  | 0.0023343358 | 0.3003663 | 90.667% | 0.31812865 | 0.0020579793 | 147/150 | 149/0/1 |
| new | new__augmented__asinh__w1__medium | asinh | augmented | medium | 1 | default | [0.0, 0.0] |  | 0.0028324343 | 0.2962963 | 89.333% | 0.31566549 | 0.002870213 | 145/150 | 147/2/1 |
| new | new__augmented__asinh__w1__medium | asinh | augmented | medium | 1 | validation_optimal | [-0.0023218531740351444, 0.0] |  | 0.0023343358 | 0.3003663 | 89.333% | 0.31566549 | 0.002870213 | 145/150 | 147/2/1 |
| new | new__augmented__asinh__w1__deeper | asinh | augmented | deeper | 1 | default | [0.0, 0.0] |  | 0.0028324343 | 0.2962963 | 90.000% | 0.31690141 | 0.0024899792 | 146/150 | 148/1/1 |
| new | new__augmented__asinh__w1__deeper | asinh | augmented | deeper | 1 | validation_optimal | [-0.0006574883341863678, 0.0] |  | 0.0023343358 | 0.3003663 | 90.000% | 0.31690141 | 0.0024899792 | 146/150 | 148/1/1 |

## Все pooled OOF-политики

| Split | Policy | Accuracy | Macro-F1 | Mean E | Mean regret | Success | SD regret |
|---|---|---|---|---|---|---|---|
| new | always_monte_carlo | 11.647% | 0.069544365 | 0.018903769 | 0.015634678 | 93.307% | 0.0033851744 |
| new | always_product_integral | 6.961% | 0.043387568 | 0.044765459 | 0.041496368 | 76.439% | 0.0072238093 |
| new | always_remizov | 81.392% | 0.29913899 | 1.1703403 | 1.1670713 | 98.527% | 2.5095338 |
| new | asinh | 80.857% | 0.2982716 | 1.1705629 | 1.1672938 | 98.126% | 2.5094102 |
| new | classifier_w1 | 81.392% | 0.29913899 | 1.1703403 | 1.1670713 | 98.527% | 2.5095338 |
| new | classifier_w12 | 30.522% | 0.23982202 | 0.016977802 | 0.013708711 | 93.173% | 0.0035518597 |
| new | classifier_w2 | 80.723% | 0.29777778 | 1.1705129 | 1.1672438 | 98.394% | 2.5096096 |
| new | classifier_w3 | 79.786% | 0.32609883 | 1.1709416 | 1.1676726 | 97.992% | 2.5094419 |
| new | classifier_w5 | 72.691% | 0.37339856 | 0.044448116 | 0.041179025 | 97.323% | 0.085697877 |
| new | classifier_w8 | 56.627% | 0.3609834 | 0.048295604 | 0.045026513 | 95.984% | 0.08535738 |
| new | nested_selection | 74.297% | 0.35920208 | 1.1731085 | 1.1698394 | 96.653% | 2.5080992 |
| new | raw | 50.469% | 0.28176158 | 1.1771007 | 1.1738317 | 96.653% | 2.5056324 |
| old | always_monte_carlo | 12.052% | 0.071702945 | 0.017308482 | 0.014330219 | 93.974% | 0.0030070604 |
| old | always_product_integral | 6.169% | 0.038738739 | 0.045478122 | 0.04249986 | 76.471% | 0.0089409074 |
| old | always_remizov | 81.779% | 0.29992107 | 1.2536817 | 1.2507035 | 98.852% | 2.6702437 |
| old | asinh | 81.205% | 0.30626781 | 0.046110597 | 0.043132334 | 98.422% | 0.094017774 |
| old | classifier_w1 | 81.779% | 0.29992107 | 1.2536817 | 1.2507035 | 98.852% | 2.6702437 |
| old | classifier_w12 | 33.429% | 0.28411917 | 0.017039879 | 0.014061617 | 93.400% | 0.0023190709 |
| old | classifier_w2 | 81.636% | 0.31450584 | 1.2537831 | 1.2508048 | 98.709% | 2.6701555 |
| old | classifier_w3 | 80.631% | 0.32998425 | 1.2539877 | 1.2510095 | 98.565% | 2.6701232 |
| old | classifier_w5 | 73.601% | 0.36855572 | 0.047661345 | 0.044683083 | 97.848% | 0.093532247 |
| old | classifier_w8 | 57.532% | 0.36222062 | 0.013623386 | 0.010645124 | 95.265% | 0.0028778577 |
| old | nested_selection | 71.879% | 0.3610283 | 1.2558958 | 1.2529175 | 97.848% | 2.6689099 |
| old | raw | 49.498% | 0.28616787 | 1.2622071 | 1.2592289 | 95.839% | 2.6659151 |

Все отдельные fold и inner selection:

| Split | Fold | Policy | Chosen spec | Accuracy | Macro-F1 | Mean E | Mean regret | Success |
|---|---|---|---|---|---|---|---|---|
| old | 1 | classifier_w1 | classifier_w1 | 80.556% | 0.2974359 | 0.0038825119 | 0.0006157439 | 100.000% |
| old | 1 | classifier_w2 | classifier_w2 | 79.861% | 0.2960103 | 0.0045291347 | 0.0012623667 | 99.306% |
| old | 1 | classifier_w3 | classifier_w3 | 79.861% | 0.32349762 | 0.0046544135 | 0.0013876454 | 99.306% |
| old | 1 | classifier_w5 | classifier_w5 | 72.222% | 0.3918041 | 0.0069765847 | 0.0037098166 | 97.917% |
| old | 1 | classifier_w8 | classifier_w8 | 54.167% | 0.37475634 | 0.015507191 | 0.012240423 | 94.444% |
| old | 1 | classifier_w12 | classifier_w12 | 30.556% | 0.27968677 | 0.021247321 | 0.017980553 | 90.972% |
| old | 1 | raw | raw | 14.583% | 0.093452003 | 0.018994868 | 0.0157281 | 93.056% |
| old | 1 | asinh | asinh | 79.861% | 0.2960103 | 0.0043556809 | 0.0010889129 | 99.306% |
| old | 1 | always_remizov | always_remizov | 80.556% | 0.2974359 | 0.0038825119 | 0.0006157439 | 100.000% |
| old | 1 | always_monte_carlo | always_monte_carlo | 14.583% | 0.084848485 | 0.01628669 | 0.013019922 | 92.361% |
| old | 1 | always_product_integral | always_product_integral | 4.861% | 0.030905077 | 0.060357318 | 0.05709055 | 67.361% |
| old | 1 | nested_selection | always_remizov | 80.556% | 0.2974359 | 0.0038825119 | 0.0006157439 | 100.000% |
| old | 2 | classifier_w1 | classifier_w1 | 77.857% | 0.291834 | 0.003154548 | 0.00051931935 | 100.000% |
| old | 2 | classifier_w2 | classifier_w2 | 78.571% | 0.32199626 | 0.0030497103 | 0.00041448172 | 100.000% |
| old | 2 | classifier_w3 | classifier_w3 | 77.857% | 0.31927575 | 0.0036318095 | 0.00099658094 | 99.286% |
| old | 2 | classifier_w5 | classifier_w5 | 75.714% | 0.43932169 | 0.0052895289 | 0.0026543003 | 97.857% |
| old | 2 | classifier_w8 | classifier_w8 | 62.143% | 0.44565108 | 0.010211521 | 0.0075762921 | 95.000% |
| old | 2 | classifier_w12 | classifier_w12 | 40.714% | 0.36420225 | 0.015165357 | 0.012530128 | 92.857% |
| old | 2 | raw | raw | 15.714% | 0.093418259 | 0.026174442 | 0.023539213 | 93.571% |
| old | 2 | asinh | asinh | 76.429% | 0.28879892 | 0.0043772471 | 0.0017420185 | 99.286% |
| old | 2 | always_remizov | always_remizov | 77.857% | 0.291834 | 0.003154548 | 0.00051931935 | 100.000% |
| old | 2 | always_monte_carlo | always_monte_carlo | 15.714% | 0.090534979 | 0.017054882 | 0.014419653 | 94.286% |
| old | 2 | always_product_integral | always_product_integral | 6.429% | 0.040268456 | 0.042853146 | 0.040217918 | 75.000% |
| old | 2 | nested_selection | classifier_w1 | 61.429% | 0.36478203 | 0.0093522331 | 0.0067170045 | 95.714% |
| old | 3 | classifier_w1 | classifier_w1 | 82.143% | 0.30065359 | 6.024167 | 6.0207192 | 97.143% |
| old | 3 | classifier_w2 | classifier_w2 | 81.429% | 0.33815749 | 6.0241112 | 6.0206635 | 97.143% |
| old | 3 | classifier_w3 | classifier_w3 | 79.286% | 0.35480309 | 6.02426 | 6.0208122 | 97.143% |
| old | 3 | classifier_w5 | classifier_w5 | 70.714% | 0.33075803 | 0.012512495 | 0.0090647302 | 96.429% |
| old | 3 | classifier_w8 | classifier_w8 | 56.429% | 0.30252713 | 0.018245064 | 0.0147973 | 94.286% |
| old | 3 | classifier_w12 | classifier_w12 | 35.000% | 0.25017253 | 0.017660158 | 0.014212394 | 91.429% |
| old | 3 | raw | raw | 81.429% | 0.2992126 | 6.0251573 | 6.0217096 | 96.429% |
| old | 3 | asinh | asinh | 82.143% | 0.30183727 | 0.0098514214 | 0.006403657 | 97.143% |
| old | 3 | always_remizov | always_remizov | 82.143% | 0.30065359 | 6.024167 | 6.0207192 | 97.143% |
| old | 3 | always_monte_carlo | always_monte_carlo | 10.000% | 0.060606061 | 0.022501779 | 0.019054015 | 92.857% |
| old | 3 | always_product_integral | always_product_integral | 7.857% | 0.048565121 | 0.039995261 | 0.036547496 | 75.000% |
| old | 3 | nested_selection | classifier_w3 | 82.143% | 0.37420688 | 6.0240969 | 6.0206491 | 97.143% |
| old | 4 | classifier_w1 | classifier_w1 | 83.704% | 0.30376344 | 0.2147236 | 0.21211025 | 97.778% |
| old | 4 | classifier_w2 | classifier_w2 | 83.704% | 0.30376344 | 0.2147236 | 0.21211025 | 97.778% |
| old | 4 | classifier_w3 | classifier_w3 | 82.222% | 0.33758349 | 0.21487015 | 0.2122568 | 97.778% |
| old | 4 | classifier_w5 | classifier_w5 | 74.074% | 0.32583904 | 0.21584356 | 0.21323021 | 97.778% |
| old | 4 | classifier_w8 | classifier_w8 | 57.778% | 0.36590595 | 0.011764407 | 0.0091510557 | 95.556% |
| old | 4 | classifier_w12 | classifier_w12 | 33.333% | 0.32 | 0.015555344 | 0.012941993 | 94.815% |
| old | 4 | raw | raw | 82.963% | 0.3022942 | 0.21498776 | 0.21237441 | 97.778% |
| old | 4 | asinh | asinh | 82.963% | 0.3022942 | 0.21520658 | 0.21259323 | 97.037% |
| old | 4 | always_remizov | always_remizov | 83.704% | 0.30376344 | 0.2147236 | 0.21211025 | 97.778% |
| old | 4 | always_monte_carlo | always_monte_carlo | 10.370% | 0.062639821 | 0.016909851 | 0.0142965 | 93.333% |
| old | 4 | always_product_integral | always_product_integral | 5.926% | 0.037296037 | 0.037034573 | 0.034421222 | 77.778% |
| old | 4 | nested_selection | classifier_w12 | 84.444% | 0.3494377 | 0.21465501 | 0.21204166 | 97.778% |
| old | 5 | classifier_w1 | classifier_w1 | 84.783% | 0.30588235 | 0.0032202297 | 0.00031433852 | 99.275% |
| old | 5 | classifier_w2 | classifier_w2 | 84.783% | 0.30588235 | 0.0032202297 | 0.00031433852 | 99.275% |
| old | 5 | classifier_w3 | classifier_w3 | 84.058% | 0.30446194 | 0.0032383052 | 0.00033241404 | 99.275% |
| old | 5 | classifier_w5 | classifier_w5 | 75.362% | 0.33102837 | 0.0042330865 | 0.0013271953 | 99.275% |
| old | 5 | classifier_w8 | classifier_w8 | 57.246% | 0.3007216 | 0.012248898 | 0.0093430069 | 97.101% |
| old | 5 | classifier_w12 | classifier_w12 | 27.536% | 0.18249889 | 0.015374191 | 0.0124683 | 97.101% |
| old | 5 | raw | raw | 55.072% | 0.27098727 | 0.0058935412 | 0.00298765 | 98.551% |
| old | 5 | asinh | asinh | 84.783% | 0.35010979 | 0.0033838132 | 0.00047792202 | 99.275% |
| old | 5 | always_remizov | always_remizov | 84.783% | 0.30588235 | 0.0032202297 | 0.00031433852 | 99.275% |
| old | 5 | always_monte_carlo | always_monte_carlo | 9.420% | 0.057395143 | 0.013753378 | 0.010847487 | 97.101% |
| old | 5 | always_product_integral | always_product_integral | 5.797% | 0.03652968 | 0.046437342 | 0.043531451 | 87.681% |
| old | 5 | nested_selection | classifier_w12 | 50.725% | 0.2748538 | 0.0082533245 | 0.0053474333 | 98.551% |
| new | 1 | classifier_w1 | classifier_w1 | 83.221% | 0.3028083 | 0.19484823 | 0.19112802 | 97.987% |
| new | 1 | classifier_w2 | classifier_w2 | 81.879% | 0.300123 | 0.19541909 | 0.19169889 | 97.315% |
| new | 1 | classifier_w3 | classifier_w3 | 81.208% | 0.36173768 | 0.19621891 | 0.1924987 | 96.644% |
| new | 1 | classifier_w5 | classifier_w5 | 69.799% | 0.33671611 | 0.19830782 | 0.19458761 | 95.302% |
| new | 1 | classifier_w8 | classifier_w8 | 51.678% | 0.32753091 | 0.20147833 | 0.19775813 | 93.960% |
| new | 1 | classifier_w12 | classifier_w12 | 29.530% | 0.22317978 | 0.017863847 | 0.014143642 | 91.275% |
| new | 1 | raw | raw | 81.879% | 0.300123 | 0.19511546 | 0.19139526 | 97.987% |
| new | 1 | asinh | asinh | 82.550% | 0.30147059 | 0.19528582 | 0.19156562 | 97.315% |
| new | 1 | always_remizov | always_remizov | 83.221% | 0.3028083 | 0.19484823 | 0.19112802 | 97.987% |
| new | 1 | always_monte_carlo | always_monte_carlo | 9.396% | 0.057259714 | 0.018737695 | 0.01501749 | 91.946% |
| new | 1 | always_product_integral | always_product_integral | 7.383% | 0.045833333 | 0.052848066 | 0.04912786 | 71.812% |
| new | 1 | nested_selection | classifier_w8 | 57.047% | 0.34911097 | 0.20431242 | 0.20059222 | 91.275% |
| new | 2 | classifier_w1 | classifier_w1 | 79.866% | 0.2960199 | 0.0035127348 | 0.00066883289 | 100.000% |
| new | 2 | classifier_w2 | classifier_w2 | 79.866% | 0.2960199 | 0.0035127348 | 0.00066883289 | 100.000% |
| new | 2 | classifier_w3 | classifier_w3 | 81.208% | 0.3527044 | 0.0032866809 | 0.00044277903 | 100.000% |
| new | 2 | classifier_w5 | classifier_w5 | 75.839% | 0.47203179 | 0.0063327762 | 0.0034888743 | 99.329% |
| new | 2 | classifier_w8 | classifier_w8 | 59.732% | 0.42603853 | 0.0091182482 | 0.0062743463 | 97.987% |
| new | 2 | classifier_w12 | classifier_w12 | 28.859% | 0.25231849 | 0.019398594 | 0.016554692 | 91.946% |
| new | 2 | raw | raw | 65.101% | 0.30111111 | 0.0045354721 | 0.0016915702 | 99.329% |
| new | 2 | asinh | asinh | 79.195% | 0.29463171 | 0.0035669509 | 0.00072304907 | 99.329% |
| new | 2 | always_remizov | always_remizov | 79.866% | 0.2960199 | 0.0035127348 | 0.00066883289 | 100.000% |
| new | 2 | always_monte_carlo | always_monte_carlo | 14.765% | 0.085769981 | 0.01395122 | 0.011107319 | 93.960% |
| new | 2 | always_product_integral | always_product_integral | 5.369% | 0.033970276 | 0.042775577 | 0.039931675 | 77.181% |
| new | 2 | nested_selection | classifier_w5 | 80.537% | 0.39250752 | 0.0056908852 | 0.0028469833 | 99.329% |
| new | 3 | classifier_w1 | classifier_w1 | 81.879% | 0.300123 | 5.6607936 | 5.6569839 | 97.315% |
| new | 3 | classifier_w2 | classifier_w2 | 79.866% | 0.2960199 | 5.6610877 | 5.657278 | 97.315% |
| new | 3 | classifier_w3 | classifier_w3 | 79.195% | 0.29463171 | 5.6612115 | 5.6574018 | 97.315% |
| new | 3 | classifier_w5 | classifier_w5 | 71.141% | 0.4017636 | 0.0059264751 | 0.0021167515 | 97.987% |
| new | 3 | classifier_w8 | classifier_w8 | 59.060% | 0.35610377 | 0.0090774876 | 0.0052677641 | 96.644% |
| new | 3 | classifier_w12 | classifier_w12 | 29.530% | 0.23992979 | 0.01411979 | 0.010310066 | 94.631% |
| new | 3 | raw | raw | 81.879% | 0.300123 | 5.6607936 | 5.6569839 | 97.315% |
| new | 3 | asinh | asinh | 81.879% | 0.300123 | 5.6607936 | 5.6569839 | 97.315% |
| new | 3 | always_remizov | always_remizov | 81.879% | 0.300123 | 5.6607936 | 5.6569839 | 97.315% |
| new | 3 | always_monte_carlo | always_monte_carlo | 10.738% | 0.064646465 | 0.024089493 | 0.020279769 | 92.617% |
| new | 3 | always_product_integral | always_product_integral | 7.383% | 0.045833333 | 0.035867598 | 0.032057874 | 75.168% |
| new | 3 | nested_selection | asinh | 81.879% | 0.300123 | 5.6607936 | 5.6569839 | 97.315% |
| new | 4 | classifier_w1 | classifier_w1 | 77.931% | 0.29198966 | 0.0043127182 | 0.0018029277 | 98.621% |
| new | 4 | classifier_w2 | classifier_w2 | 77.931% | 0.29198966 | 0.0043127182 | 0.0018029277 | 98.621% |
| new | 4 | classifier_w3 | classifier_w3 | 74.483% | 0.30888791 | 0.0055115691 | 0.0030017785 | 97.241% |
| new | 4 | classifier_w5 | classifier_w5 | 71.034% | 0.31630759 | 0.006647315 | 0.0041375245 | 95.862% |
| new | 4 | classifier_w8 | classifier_w8 | 61.379% | 0.34414124 | 0.0087263824 | 0.0062165919 | 94.483% |
| new | 4 | classifier_w12 | classifier_w12 | 37.931% | 0.26195978 | 0.012233312 | 0.0097235217 | 93.103% |
| new | 4 | raw | raw | 14.483% | 0.08815248 | 0.020009248 | 0.017499457 | 93.103% |
| new | 4 | asinh | asinh | 76.552% | 0.29019608 | 0.0049537019 | 0.0024439114 | 97.931% |
| new | 4 | always_remizov | always_remizov | 77.931% | 0.29198966 | 0.0043127182 | 0.0018029277 | 98.621% |
| new | 4 | always_monte_carlo | always_monte_carlo | 13.793% | 0.080808081 | 0.017065156 | 0.014555365 | 92.414% |
| new | 4 | always_product_integral | always_product_integral | 8.276% | 0.050955414 | 0.040287387 | 0.037777596 | 76.552% |
| new | 4 | nested_selection | classifier_w12 | 72.414% | 0.33751205 | 0.0056419751 | 0.0031321846 | 97.241% |
| new | 5 | classifier_w1 | classifier_w1 | 83.871% | 0.30409357 | 0.0039023603 | 0.00046758125 | 98.710% |
| new | 5 | classifier_w2 | classifier_w2 | 83.871% | 0.30409357 | 0.0039023603 | 0.00046758125 | 98.710% |
| new | 5 | classifier_w3 | classifier_w3 | 82.581% | 0.30153121 | 0.0041766229 | 0.0007418439 | 98.710% |
| new | 5 | classifier_w5 | classifier_w5 | 75.484% | 0.32961973 | 0.0055767014 | 0.0021419224 | 98.065% |
| new | 5 | classifier_w8 | classifier_w8 | 51.613% | 0.33769063 | 0.013419709 | 0.0099849295 | 96.774% |
| new | 5 | classifier_w12 | classifier_w12 | 27.097% | 0.21181873 | 0.020984743 | 0.017549964 | 94.839% |
| new | 5 | raw | raw | 9.677% | 0.058823529 | 0.02055926 | 0.017124481 | 95.484% |
| new | 5 | asinh | asinh | 83.871% | 0.30409357 | 0.0039023603 | 0.00046758125 | 98.710% |
| new | 5 | always_remizov | always_remizov | 83.871% | 0.30409357 | 0.0039023603 | 0.00046758125 | 98.710% |
| new | 5 | always_monte_carlo | always_monte_carlo | 9.677% | 0.058823529 | 0.02055926 | 0.017124481 | 95.484% |
| new | 5 | always_product_integral | always_product_integral | 6.452% | 0.04040404 | 0.051651174 | 0.048216394 | 81.290% |
| new | 5 | nested_selection | classifier_w8 | 79.355% | 0.32909091 | 0.0048078459 | 0.0013730668 | 98.065% |

cv_folds.json дополнительно сохраняет все inner candidates выбранной nested политики; cv_inner_predictions.csv и policy_search/ позволяют независимо сверить выбранные biases. CV не меняет основной выбор latest Validation.
