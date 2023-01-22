"""Utilities module including useful tools"""
__all__ = ["circular_convolve"]

import warnings
import numpy as np


def circular_convolve(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Perform convolution on an array

    Parameters
    ----------
    arr : NumPy ndarray
    kernel : NumPy ndarray
    """
    k = kernel.shape[0]

    if k % 2 == 0:
        warnings.warn("Convolution kernel size is an even, so result is shifted.")

    tip = np.concatenate([arr[-(k - 1) :], arr[: k - 1]])

    main = np.convolve(arr, kernel, mode="valid")
    tip = np.convolve(tip, kernel, mode="valid")

    return np.concatenate(
        [
            tip[int(k / 2) :],
            main,
            tip[: int(k / 2)],
        ]
    )
