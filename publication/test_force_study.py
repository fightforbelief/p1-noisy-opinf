import numpy as np
from force_study import observations, coefficient, fit


def test_independent_segments_and_gapped_calibration():
    t = np.arange(12000)/100
    x = 400+100*np.cos(2*np.pi*t)
    record = np.column_stack([t,x,800-x])
    for arm in ("raw","lowpass6"):
        first,load = observations(record,0,arm)
        assert len(first)==4000-224
        np.testing.assert_allclose(load,800)
        changed=record.copy()
        changed[4000:,1]+=1000
        np.testing.assert_array_equal(observations(changed,0,arm)[0],first)
        assert coefficient(first,3)<1e-8
        assert np.max(np.abs(fit(first,40,40)[-1]))<1e-4
    gapped=record.copy()
    gapped[2000:,0]+=.05
    assert len(observations(gapped,0,"raw")[0])==4000-448
