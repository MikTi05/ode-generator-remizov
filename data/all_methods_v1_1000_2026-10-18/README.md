# Production all_methods_v1: 1000 equations

Created 2026-10-18 with the unchanged project generator. Master seed: 2488;
200 structural families × 5 parameterizations. This is a separate dataset
from `../generated/`; the old 20-task set is preserved.

Required artifacts, all sharing the `all_methods_equations_v1` prefix:

- `.csv`: ordinary equation and x0/x1/y0/v0;
- `_metadata.jsonl`: safe AST, exact solution, analytic derivatives,
  certificates, seeds, family, complexity, regime and split;
- `_ml.npz`: x_grid[256], X[1000,4,256] in [a,b,c,g] order,
  y_exact/dy_exact[1000,256], y0/v0, identifiers and split; allow_pickle=False;
- `_report.json`: canonical deterministic statistics and artifact hashes.

Each L1–L4 contains 250 equations (50 families); each of constant,
slow_varying, moderate_varying, oscillatory, high_contrast contains 200
(40 families). Each complexity/regime cell has 50 equations in 10 families.
Split by family: train 700/140, validation 150/30, test 150/30
(equations/families). No family crosses partitions.

The original equations and all coefficient ranges are preserved:
a*y'' + b*y' + c*y = g, c=c_R-lambda, g_R=-g.
y0=y_exact(-1), v0=y_exact'(-1) remain in CSV, JSONL and NPZ for other
methods. Remizov uses the bounded solution on the whole real line and
the loaded remizov_problem; it does not use initial conditions.

Validated with the production validator, including byte reproducibility,
all analytic construction certificates, exact/residual/support checks,
diversity, split and NPZ reconstruction. Global bounds follow from the
analytic construction; finite-grid checks are implementation checks.
All 1000 public-loader equations additionally match NPZ vector evaluations
and satisfy the conditions used by chernov_solver.py. No ODE solver was run.
Numerical solution accuracy on these 1000 equations is not measured.

Use explicit paths; existing defaults still select the old dataset:

```python
from pathlib import Path
from all_methods_dataset_loader import load_all_methods_equations
folder = Path("data/all_methods_v1_1000_2026-10-18")
equations = load_all_methods_equations(
    folder / "all_methods_equations_v1.csv",
    folder / "all_methods_equations_v1_metadata.jsonl",
)
problem = equations[0].remizov_problem
```

For the existing example CLI, select this directory using `--input-dir`.
IDs are scoped to the seed/configuration/dataset: an ID shared with the old
20-task dataset does not establish equation identity. Keep the dataset path
or artifact hashes with future numerical results.

Validation commands, logs, per-equation checks, environment and hashes:
[results/dataset1000_2026-10-18](../../results/dataset1000_2026-10-18/REPORT.md).
