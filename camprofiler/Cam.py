__all__ = ["Cam"]

import numpy as np


class Cam:
    """The Cam object

    Parameters
    ----------
    size : int, default = 36000
        Number of sample points in cam profile
    profile : np.ndarray, default = None
        Apply a cam profile.
        Flat profile is applied by default.
    """

    def __init__(self, size: int = 36000, profile: np.ndarray = None):
        self.SIZE = size
        if profile is None:
            self.profile = np.ones((size))
        else:
            ...
            # self.profile = self.fit_profile_polynomial_with_lines(profile)

    def fit_profile_polynomial_with_points(
        self,
        profile: np.ndarray,
        degree: int = 10,
        start: float = 0.0,
        end: float = 360.0,
    ):
        """Fit a segment of the cam profile with a polynomial curve using
        points from the curve

        Parameters
        ----------
        profile : np.ndarray
            1D numpy array with points from the curve.
            The entire profile array is used to compute the polynomial.
        degree : int
            Degree of fitted polynomial.
        start : float, default = 0.0
            Angular value (0 - 360) that marks the start of the segment.
            Curve will be fitted to [start, end).
        end : float, default = 360.0
            Angular value (0 - 360) that marks the end of the segment.
            Curve will be fitted to [start, end).
        """
        if len(profile.shape) != 1:
            raise SyntaxError("Input profile dimension incorrect.")

        coefficients = np.polynomial.polynomial.polyfit(
            x=np.linspace(start=0, stop=1, num=profile.shape[0]),
            y=profile,
            deg=degree,
        )

        self.fit_profile_polynomial_with_coefficients(coefficients, start, end)

    def fit_profile_polynomial_with_coefficients(
        self, coefficients: np.ndarray, start: float = 0.0, end: float = 360.0
    ):
        """Fit a segment of the cam profile with a polynomial curve using coefficients

        Parameters
        ----------
        coefficients : np.ndarray
            1D numpy array with coefficients in increasing order of degree
        start : float, default = 0.0
            Angular value (0 - 360) that marks the start of the segment.
            Curve will be fitted to [start, end).
        end : float, default = 360.0
            Angular value (0 - 360) that marks the end of the segment.
            Curve will be fitted to [start, end).
        """
        if len(coefficients.shape) != 1:
            raise SyntaxError("Input coefficients dimension incorrect.")

        starting_index = int(start / 360 * self.SIZE)
        ending_index = int(end / 360 * self.SIZE)

        # if profile.shape[0] == ending_index - starting_index:
        #     self.profile[starting_index:ending_index] = profile
        #     return

        self.profile[starting_index:ending_index] = np.zeros(
            (ending_index - starting_index)
        )
        for x in range(ending_index - starting_index):
            scaled_x = (
                float(x) / (ending_index - starting_index)
                + float(x + 1) / (ending_index - starting_index)
            ) / 2
            for n, c in enumerate(coefficients):
                self.profile[x + starting_index] += c * scaled_x**n
