import numpy as np
import pytest

from derivative_study import feature_jet, features, observed, parameters, time_derivatives


@pytest.mark.parametrize("system", ["burgers","allen_cahn"])
def test_pde_first_and_second_derivatives(system):
    n, nu, _, _ = parameters(system)
    k = 2*np.pi
    x = np.arange(n)*k/n
    u = 0.4*np.sin(x)
    ux, uxx = 0.4*k*np.cos(x), -k*k*u
    if system=="burgers":
        first = nu*uxx-u*ux
    else:
        first = nu*uxx+3*u*(1-u*u)
    du, ddu=time_derivatives(u,system)
    np.testing.assert_allclose(du,first,atol=1e-11)
    # Finite differences of the analytical low-mode RHS along the known tangent.
    eps=1e-5
    right=time_derivatives(u+eps*first,system)[0]
    left=time_derivatives(u-eps*first,system)[0]
    np.testing.assert_allclose(ddu,(right-left)/(2*eps),atol=2e-8)


def test_quadratic_product_rules():
    gen=np.random.default_rng(299)
    q,v,a=gen.normal(size=(3,30,3))
    d1,d2=feature_jet(q,v,a)
    h=1e-4
    plus,minus=[features(q+sign*h*v+h*h*a/2) for sign in (1,-1)]
    np.testing.assert_allclose(d1,(plus-minus)/(2*h),atol=1e-7)
    np.testing.assert_allclose(d2,(plus+minus-2*features(q))/h**2,atol=6e-7)


@pytest.mark.parametrize("m",[1,2])
def test_coupled_signed_limit_from_unrelated_scalar_curve(m):
    t=np.linspace(-1,1,101)
    q=t[:,None]**4
    z=np.column_stack([q,np.ones(len(t))])
    u,r=np.linalg.qr(z)
    first,second=4*t**3,12*t**2
    j,_=np.linalg.qr(z if m==1 else np.column_stack([z,first]))
    deriv=np.column_stack([first if m==1 else second,np.zeros(len(t))])
    v=deriv-j@(j.T@deriv)
    l=np.linalg.solve(r.T,v.T).T
    lead=-m*l.T@v[:,:1]
    errors=[]
    for h in (0.02,0.01,0.005):
        w=np.column_stack([*((t-i*h)**4 for i in range(1,m+1)),np.ones(len(t))])
        actual,direct,*_=observed(u,q,((t+h)**4)[:,None],w)
        np.testing.assert_allclose(actual,direct,atol=1e-12)
        errors.append(np.linalg.norm(actual/h**(2*m)-lead)/np.linalg.norm(lead))
    assert errors[-1]<errors[1]<errors[0] and errors[-1]<0.1
