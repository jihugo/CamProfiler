__all__ = ["seamless_convolve"]

import numpy as np
import warnings
from camprofiler.protocol import *


def seamless_convolve(A: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Perform convolution on an array
    """
    k = kernel.shape[0]

    if k % 2 == 0:
        warnings.warn(
            "Convolution kernel is an even number, convolution result will be shifted."
        )

    tip = np.concatenate([A[-(k - 1) :], A[: k - 1]])

    main = np.convolve(A, kernel, mode="valid")
    tip = np.convolve(tip, kernel, mode="valid")

    return np.concatenate(
        [
            tip[int(k / 2) :],
            main,
            tip[: int(k / 2)],
        ]
    )
