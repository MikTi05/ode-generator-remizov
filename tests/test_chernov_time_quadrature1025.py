"""Independent quadrature checks only: no numerical ODE solves."""
import math
import numpy as np
import pytest
from scipy.integrate import simpson

from diagnostics.chernov_time_quadrature1025 import quadrature_check


@pytest.mark.parametrize('T',[1.,18.692237908953764])
def test_nested_nodes_signs_and_analytic_constant_linear_quadratic_moments(T):
    arrays,checks=quadrature_check(T)
    np.testing.assert_array_equal(arrays['513']['t'],arrays['1025']['t'][::2])
    for nodes in ('513','1025'):
        t,w=arrays[nodes]['t'],arrays[nodes]['weights']
        assert np.flatnonzero(w<0).tolist()==[0]
        np.testing.assert_allclose(w[0],-2*T/(3*(len(t)-1)**2),rtol=2e-15,atol=0)
        for k in range(3):
            actual=math.fsum(float(a)*float(b) for a,b in zip(w,t**k))
            np.testing.assert_allclose(actual,T**(k+1)/(k+1),rtol=5e-15,atol=0)


def test_weights_agree_with_independent_scipy_integration_on_nonpolynomial_data():
    arrays,_=quadrature_check(18.692237908953764)
    for a in arrays.values():
        t,w=a['t'],a['weights']
        values=np.column_stack((np.exp(-t),np.sin(3*t),np.cos(.7*t)))
        np.testing.assert_allclose(np.sum(w[:,None]*values,axis=0),
            simpson(values,x=t,axis=0),rtol=2e-14,atol=1e-15)
