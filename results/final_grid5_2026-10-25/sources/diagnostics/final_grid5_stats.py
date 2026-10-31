"""Predeclared reliability inference for four fresh grid-level MC trials."""
from scipy.stats import beta,binom


def reliability(successes,n=4,equations=5,target=.95):
    """Exact binomial inference; never treats solver/resource failures as errors."""
    if not 0<=successes<=n or n<1:raise ValueError('invalid binomial counts')
    tail=.05/(2*equations)
    lo=float(beta.ppf(tail,successes,n-successes+1)) if successes else 0.
    hi=float(beta.ppf(1-tail,successes+1,n-successes)) if successes<n else 1.
    p=float(binom.cdf(successes,n,target))
    return dict(successes=successes,trials=n,rate=successes/n,
        exact_family95_probability_interval=[lo,hi],equations_adjusted=equations,
        test_null=f'p_grid_success >= {target}',pvalue=p,per_equation_alpha=.05/equations,
        reject_p95=p<=.05/equations,confirm_p95=lo>=target,
        scope='independent fresh4096-path full-grid runs, fixed h/T; not a universal MC statement')
