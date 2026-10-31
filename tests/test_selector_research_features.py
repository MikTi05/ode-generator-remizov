"""Independent simple functions and invalid input for the five interactions."""
import numpy as np
import pytest

from ml.selector_research_features import FEATURE_NAMES, additional_coefficient_features


def constant_inputs():
    x = np.linspace(-1, 1, 33)
    values = np.array([np.full_like(x, value) for value in (4.0, 3.0, -9.0, 2.0)])
    return values, x


def test_constant_case_has_manual_dimensionless_ratios_and_no_proxy_defect():
    values, x = constant_inputs()
    result = additional_coefficient_features(values, x, {"unused": object()})
    assert tuple(result) == FEATURE_NAMES
    assert len(result) == 5
    assert result["research_local_peclet"] == pytest.approx(0.5)
    assert result["research_forcing_cancellation"] == pytest.approx(1.0)
    for name in ("research_forcing_curvature_ratio", "research_diffusion_gradient_ratio",
                 "research_reaction_proxy_defect"):
        assert result[name] == pytest.approx(0.0, abs=1e-12)


def test_zero_forcing_and_amplitude_sign_invariance():
    values, x = constant_inputs()
    values[3] = x * x + 1
    original = additional_coefficient_features(values, x)
    values[3] *= -17
    scaled = additional_coefficient_features(values, x)
    assert scaled == pytest.approx(original)
    values[3] = 0
    zero = additional_coefficient_features(values, x)
    assert all(np.isfinite(list(zero.values())))
    assert zero["research_forcing_curvature_ratio"] == 0
    assert zero["research_forcing_cancellation"] == 0
    assert zero["research_reaction_proxy_defect"] == 0


def test_quadratic_forcing_matches_manual_proxy_defect():
    values, x = constant_inputs()
    values[1] = 0
    values[3] = -(1 + x * x)
    result = additional_coefficient_features(values, x)
    expected = 8 / (9 * np.sqrt(np.mean((1 + x * x) ** 2)))
    assert result["research_forcing_curvature_ratio"] == pytest.approx(expected)
    assert result["research_reaction_proxy_defect"] == pytest.approx(expected)


@pytest.mark.parametrize("change", ["nan", "diffusion", "reaction", "grid", "interval", "shape"])
def test_invalid_coefficient_input_is_rejected(change):
    values, x = constant_inputs()
    if change == "nan":
        values[3, 2] = np.nan
    elif change == "diffusion":
        values[0, 2] = 0
    elif change == "reaction":
        values[2, 2] = 1
    elif change == "grid":
        x[3] = x[2]
    elif change == "interval":
        x = x + 1
    else:
        values = values[:3]
    with pytest.raises(ValueError):
        additional_coefficient_features(values, x)
