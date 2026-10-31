# All-methods dataset v1 for the Remizov solver

## Purpose and versioning

This independent project preserves the existing `all_methods_v1`
manufactured-solution generator and the Remizov/Chernoff solver.
Versioned artifact names and schemas are unchanged. Generation constructs
and validates equations without running a numerical ODE solver.

## One equation in two equivalent forms

The ordinary IVP on the fixed interval `[-1, 1]` is

\[
 a(x)y''+b(x)y'+c(x)y=g(x),\qquad
 y(-1)=y_0,\quad y'(-1)=v_0.
\]

The same equation is presented to the Remizov solver as

\[
 a(x)y''+b(x)y'+(c_R(x)-\lambda)y=-g_R(x),
\]

with the identities

\[
 c=c_R-\lambda,\qquad g_R=-g.
\]

The ordinary and Remizov callables are reconstructed from the same validated
AST and construction parameters, rather than independently parsing two sets
of formulas.  This prevents sign drift between the two representations.

## Constructive certificates

Four globally safe raw expression trees are denoted by
`T_a`, `T_b`, `T_c`, and `T_y`.  For sampled finite constants
`A_min`, `A_max`, `B_scale`, `C_scale`, and `gap`, the coefficients are

\[
\begin{aligned}
A_{mid}&=(A_{max}+A_{min})/2,\\
A_{half}&=(A_{max}-A_{min})/2,\\
a(x)&=A_{mid}+A_{half}\tanh(T_a(x)),\\
b(x)&=B_{scale}\tanh(T_b(x)),\\
c_R(x)&=C_{scale}\tanh(T_c(x)),\\
\lambda&=C_{scale}+gap,\\
c(x)&=c_R(x)-\lambda.
\end{aligned}
\]

Consequently, without relying on a finite grid,

\[
A_{min}\le a\le A_{max},\quad
|b|\le B_{scale},\quad
|c_R|\le C_{scale},\quad
\lambda-\|c_R\|_\infty\ge gap>0,
\]

and `c(x) <= -gap < 0`.  The stored bounds are analytic construction
certificates.  Dense-grid checks are implementation smoke tests only and are
never described as proofs of global bounds.

## Globally safe expression grammar

Raw trees contain only the following typed operations:

- primitives: `constant`, `sin_affine`, `cos_affine`, `tanh_affine`,
  `gaussian`, `lorentz`, and `saturating`;
- unary composition: `sin`, `cos`, `tanh`, `exp_neg_square`, and
  `rational_saturate`;
- binary composition: finite `lincomb` and `mul`.

Here `gaussian` means `exp(-alpha*(x-mu)^2)`, `lorentz` means
`1/(1+alpha*(x-mu)^2)`, and `saturating` means
`(x-mu)/sqrt(1+(x-mu)^2)`.  Only `alpha` in `gaussian` and `lorentz` is
required to be positive.  Affine frequencies and `lincomb` weights may be
signed; all numerical parameters must be finite and lie in the generator's
regime-specific ranges.
Every primitive is globally defined, bounded, and smooth.  The permitted
compositions, finite linear combinations, and products preserve these
properties and bounded derivatives.

There are no generic polynomial leaves, `log`, unrestricted `sqrt`, generic
division, `exp(u)`, `tan`, `abs`, `sign`, `min/max`, or discontinuous
coefficient `Piecewise` nodes.  A strict AST decoder rejects unknown
operations, wrong arity, extra/missing parameters, non-finite parameters, and
invalid positive scales.

## Exact compact-support solution

For

\[
z=(x-\mu)/R,
\]

the normalized smooth bump is

\[
\phi(z)=
\begin{cases}
\exp\!\left(1-\frac{1}{1-z^2}\right),&|z|<1,\\
0,&|z|\ge1.
\end{cases}
\]

The generator samples `mu` in `[-0.25, 0.25]` and `R` in `[1.5, 3.0]`.
These ranges guarantee that the certified interval `[mu-R, mu+R]` strictly
contains the entire working interval `[-1, 1]` and continues beyond both
endpoints.  This avoids the earlier degeneracy in which almost every initial
state was zero.

The manufactured solution is

\[
y_{exact}(x)=solution\_scale\,\phi((x-\mu)/R)T_y(x).
\]

The bump and expression tree are differentiated analytically by a recursive
second-order jet evaluator.  Finite differences are never used to construct
`y'`, `y''`, or the right-hand side.  A raw solution is rejected if its
sampled maximum is below `1e-3`; otherwise one common scale normalizes its
sampled maximum absolute value to a target in `[0.5, 1.5]`.

Initial data are consequences of the same exact solution:

\[
y_0=y_{exact}(-1),\qquad v_0=y'_{exact}(-1).
\]

Finally,

\[
g=ay''_{exact}+by'_{exact}+cy_{exact},\qquad g_R=-g.
\]

Since the bump and all its derivatives vanish outside the certified interval,
`y_exact`, its derivatives, `g`, and `g_R` are all zero outside that same
interval.  Their minimal mathematical supports may be smaller if `T_y` or a
coefficient combination introduces additional zeros.  The implementation
evaluates only the interior bump branch, so the certified boundary and
exterior values are exactly zero in floating-point arithmetic.

## Families, complexity, regimes, and split

Production v1 contains 200 structural families with five numerical
parameterizations per family.  The topology of all four trees is fixed within
a family; amplitudes, frequencies, phases, centers, widths, coefficient
scales, gap, and solution scale change.  A canonical SHA-256 structure
signature contains operator topology but no numerical parameters.

The 200 families are balanced over 20 cells formed by four complexity levels
`L1`--`L4` and five regimes: `constant`, `slow_varying`,
`moderate_varying`, `oscillatory`, and `high_contrast`.  Each cell has ten
families and therefore 50 equations.  In the `constant` regime, `a`, `b`, and
`c_R` are constant while the exact solution may retain the selected structural
complexity.

Splitting is performed before sample generation and only at `family_id`
level.  Production allocation is 140/30/30 families, hence 700/150/150
equations in train/validation/test.  No family can occur in two splits.  This
group split prevents near-clone parameterizations from leaking across
partitions.

## Reproducibility and validation

The default master seed is 2488.  Family, topology, sample, and retry streams
use addressable `numpy.random.SeedSequence` paths and `Generator(PCG64)`;
they do not use global NumPy state, `random.Random`, or Python's randomized
`hash()`.  A rejection in one sample therefore cannot perturb another
sample's random stream.

With the production configuration, generation checks each candidate on a
2049-point grid over `[-1, 1]`.  The standalone full artifact validator also
performs an auxiliary 1025-point smoke check over `[-10, 10]`.  The latter is
not a proof of global bounds; those bounds follow from the constructive
parameterization above.  Checks cover finite values, the configurable broad
`1e6` emergency safety bound, scale-aware residual, both sign
identities, analytic certificates, exact exterior zeros, initial-condition
consistency, and serialization round trips.

Every rejected attempt has a classified reason and deterministic attempt
seed in the in-memory generation result.  The canonical JSON report persists
the total and counts by rejection reason; the four required v1 artifacts do
not currently include a separate rejection-detail JSONL file.

## Files and sampled numerical representation

- `all_methods_equations_v1.csv` contains the nine ordinary-IVP columns used
  by `CauchyProblem`.
- `all_methods_equations_v1_metadata.jsonl` contains certificates, exact
  formulas, seeds, family/split metadata, and complete serialized trees.
- `all_methods_equations_v1_ml.npz` stores a fixed 256-point grid, channels in
  strict order `[a,b,c,g]`, exact trajectories, initial conditions, and
  Unicode identifiers without pickle-dependent object arrays.
- `all_methods_equations_v1_report.json` contains deterministic generation and
  validation statistics.

The human-readable formula strings include declarative `bump`, `bump_d1`,
and `bump_d2` tokens. They are never executed. The safe loader reconstructs
the authoritative v1 AST, verifies canonical formula serialization, and
returns `CauchyProblem` from `src/problem_types.py`, `RemizovProblem`, and
exact-solution evaluators.

CSV plus the matching metadata JSONL is required for executable equations
and exact errors on arbitrary target grids. The full artifact validator
requires all four files and regenerates the dataset from its seed/config.
The `_ml.npz` filename, `ml_grid_size`, split fields, and sampled arrays are
retained parts of the versioned data contract; no ML model is included.

## Solver interpretation

The solver computes a global resolvent approximation. It does not accept
arbitrary initial conditions. In this dataset `y0` and `v0` are consequences
of the same smooth compactly supported exact solution that defines `g`.
Pass the loaded `remizov_problem` directly, without changing the sign of
`g_remizov` or substituting ordinary `c` for `c_remizov`.

The solver returns finite-n values in `result.f`, without a derivative array.
Successful quadrature/spatial refinement does not certify total solution
error. Preserve and report failures instead of silently increasing budgets.
The truncation formula is unchanged and does not incorporate the norm of
`g_remizov`; interpreting `tail_eps` as a universal absolute error bound
requires a separate mathematical review.
