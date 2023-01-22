"""Utilities module including useful tools"""
__all__ = ["circular_convolve", "circular_linear_interp"]

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


def circular_linear_interp(arr: np.ndarray, new_size: int) -> np.ndarray:
    """Perform linear interpolation

    The first data point in arr is

    Parameters
    ----------
    arr : 1D NumPy ndarray
    new_size : int
    """

    arr_modified = np.append(arr, arr[0])
    x_arr_curr = np.arange(arr.shape[0] + 1)
    x_arr_new = np.linspace(start=0, stop=arr.shape[0], num=new_size + 1)
    return np.interp(x_arr_new, x_arr_curr, arr_modified)[:-1]
