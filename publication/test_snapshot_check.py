import numpy as np
from snapshot_check import differentiate, ANCHORS, DELTA_TICKS, TICKS


def test_stencil_quadratic_and_disjoint_acquisition_times():
    for h in (.001,.05):
        t=np.array([.05,.43,.70])
        frames=[(t+j*h)**2+3*t+1 for j in range(4)]
        first,second=differentiate(frames,h)
        np.testing.assert_allclose(first,2*t,rtol=1e-10)
        np.testing.assert_allclose(second,2.,rtol=1e-8)
    extra=(ANCHORS[:,None]+DELTA_TICKS*np.arange(1,4)).ravel()
    assert not np.intersect1d(extra,TICKS).size
