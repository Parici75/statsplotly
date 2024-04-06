"""Utility data functions."""

from typing import Any

import numpy as np
from numpy.typing import NDArray


def rand_jitter(arr: NDArray[Any], jitter_amount: float = 1) -> NDArray[Any]:
    """from https://stackoverflow.com/questions/8671808/matplotlib-avoiding
    -overlapping-datapoints-in-a-scatter-dot-beeswarm-plot"""
    spread = 0.01 * (max(arr) - min(arr)) * jitter_amount or 0.05 * jitter_amount

    return arr + np.random.randn(len(arr)) * spread
