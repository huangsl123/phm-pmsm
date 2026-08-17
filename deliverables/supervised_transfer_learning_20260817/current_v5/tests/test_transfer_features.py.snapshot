import numpy as np

from data.transfer_features import dimensionless_features


def test_dimensionless_features_are_finite_and_scale_invariant():
    rng = np.random.default_rng(17)
    window = rng.normal(size=(4096, 3)).astype(np.float32)
    original = dimensionless_features(window, 10_000)
    scaled = dimensionless_features(window * 7.5, 10_000)
    assert original.shape == (376,)
    assert np.isfinite(original).all()
    np.testing.assert_allclose(original, scaled, rtol=2e-5, atol=2e-5)
