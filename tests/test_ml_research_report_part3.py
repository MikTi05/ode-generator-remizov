"""Report integrity checks, using saved evidence only."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('part3_check', ROOT / 'diagnostics/check_part3_report.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def test_report_matches_saved_evidence_and_compiles_cleanly():
    result = checker.audit()
    assert result['passed'] and result['protected_files'] >= 1584
    assert result['solver_calls'] == result['new_training'] == result['new_model_inference'] == 0


def test_reported_decision_rule_boundary_and_MC_eligibility():
    # Unit fixtures check the algebraic rule; they are not research observations.
    def estimates(p128, p256, pmc, t128=10, t256=20, tmc=9):
        return {k: {'probability_accuracy': p, 'predicted_full_warm_seconds': t}
                for k, p, t in [('R128', p128, t128), ('R256', p256, t256), ('MC1', pmc, tmc)]}
    assert checker.stated_rule(estimates(.9, .9, .9)) == 'MC1'  # exact 10% boundary
    assert checker.stated_rule(estimates(.9, .9, .9, tmc=9.000001)) == 'R128'
    assert checker.stated_rule(estimates(.9, .9, .8, tmc=1)) == 'R128'  # too low probability
    assert checker.stated_rule(estimates(.8, .8, .9, tmc=100)) == 'MC1'  # no eligible R
    assert checker.stated_rule(estimates(.3, .3, .1)) == 'R128'  # no absolute q=.95 gate
