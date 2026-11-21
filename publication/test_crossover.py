import numpy as np
from crossover import rhs,second,rk4,scales


def test_lorenz96_tangent_flow_and_orientation():
    x=np.random.default_rng(7).normal(size=(2,40))
    v=rhs(x)
    manual=np.array([[(row[(i+1)%40]-row[i-2])*row[i-1]-row[i]+8 for i in range(40)] for row in x])
    np.testing.assert_allclose(manual,v)
    h=1e-5
    np.testing.assert_allclose((rhs(x+h*v)-rhs(x-h*v))/(2*h),second(x,v),atol=1e-8)
    np.testing.assert_allclose((rk4(x,h)-2*x+rk4(x,-h))/h**2,second(x,v),atol=2e-5)


def test_crossover_scalars():
    for h,t in [(.002,.002),(.002,.02),(.002,.0002)]:
        assert np.isclose(scales(h,t,1)[0],h*t)
        joint,sep=scales(h,t,2)
        assert np.isclose(joint/sep,1+3*h/t)
