import numpy as np

from qnav.util.transformations import quaternion_to_euler, quaternion_to_euler_vec


def test_quaternion_to_euler_clips_pitch_roundoff():
    component = np.sqrt(0.5) * (1.0 + 1e-15)
    quaternion = np.array([component, 0.0, component, 0.0])

    scalar = quaternion_to_euler(quaternion)
    vector = quaternion_to_euler_vec(quaternion[None, :])[0]

    assert np.isfinite(scalar).all()
    np.testing.assert_allclose(scalar[1], 90.0, atol=1e-12)
    np.testing.assert_allclose(vector, scalar, atol=1e-12)
