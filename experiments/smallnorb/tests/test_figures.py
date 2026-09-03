import numpy as np
import pytest

from experiments.smallnorb.figures import _normalize_trajectory_by_max


def test_normalize_trajectory_by_its_own_maximum():
    normalized = _normalize_trajectory_by_max([2.0, 8.0, np.nan, 4.0])

    np.testing.assert_allclose(
        normalized,
        np.asarray([0.25, 1.0, np.nan, 0.5]),
        equal_nan=True,
    )


def test_normalize_trajectory_rejects_nonpositive_maximum():
    with pytest.raises(ValueError, match="must be positive"):
        _normalize_trajectory_by_max([0.0, -1.0])
