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
            self.profile = self.fit_with_straight_lines(profile)

    def fit_profile_with_straight_lines(
        self,
        profile: np.ndarray,
        start: float = 0.0,
        end: float = 360.0,
        smoothen: bool = True,
    ):
        """Fit a segment of the cam profile with a straight lines defined by profile

        Parameters
        ----------
        profile : np.ndarray
            1D Numpy array or n x 2 Numpy array that define points.
            1D array represents evenly-spaced cam lift values
            n x 2 array contains angle in the first column and lift values in second
                column, these points must be in order with increasing angle.

        start : float, default = 0.0
            Angular value (0 - 360) that marks the start of the segment.
            Curve will be fitted to [start, end).

        end : float, default = 360.0
            Angular value (0 - 360) that marks the end of the segment.
            Curve will be fitted to [start, end).

        smoothen : bool, default = True
            Smoothen the entire profile after fitting.
        """

        if len(profile.shape) != 1:
            if profile.shape[1] != 2:
                raise SyntaxError("Input profile has incorrect format.")
            length = profile.shape[0]

        # convert 1D into 2D
        else:
            length = profile.shape[0]
            profile = np.array([np.arange(start=0, stop=length), profile]).transpose()

        starting_index = int(start / 360 * self.SIZE)
        ending_index = int(end / 360 * self.SIZE)

        idx1 = (
            starting_index
            + int(profile[0][0] / (length - 1) * (ending_index - starting_index) + 0.5)
            + 1
        )
        dummy_profile = np.concatenate([self.profile, [0]])
        for i in range(length - 1):
            curr_point = profile[i]
            next_point = profile[i + 1]

            idx0 = idx1 - 1
            idx1 = starting_index + int(
                next_point[0] / (length - 1) * (ending_index - starting_index) + 0.5
            )

            dummy_profile[idx0:idx1] = np.linspace(
                start=curr_point[1], stop=next_point[1], num=idx1 - idx0
            )
        self.profile = dummy_profile[:-1]

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
        self,
        coefficients: np.ndarray,
        cam_start: float = 0.0,
        cam_end: float = 360.0,
        x_start: int = 0,
        x_end: int = 1,
    ):
        """Fit a segment of the cam profile with a polynomial curve using coefficients

        Parameters
        ----------
        coefficients : np.ndarray
            1D numpy array with coefficients in increasing order of degree
        cam_start : float, default = 0.0
            Angular value (0 - 360) that marks the start of the segment.
            Curve will be fitted to [start, end).
        cam_end : float, default = 360.0
            Angular value (0 - 360) that marks the end of the segment.
            Curve will be fitted to [start, end).
        x_start : int, default = 0
            For polynomial with variable x, the left bound on x
        x_end : int, default = 1
            For polynomial with variable x, the right bound on x
        """
        if len(coefficients.shape) != 1:
            raise SyntaxError("Input coefficients dimension incorrect.")

        starting_index = int(cam_start / 360 * self.SIZE)
        ending_index = int(cam_end / 360 * self.SIZE)

        self.profile[starting_index:ending_index] = np.zeros(
            (ending_index - starting_index)
        )
        x_range = x_end - x_start
        for x in range(ending_index - starting_index):
            scaled_x = (
                (
                    float(x) / (ending_index - starting_index)
                    + float(x + 1) / (ending_index - starting_index)
                )
                / 2
            ) * x_range + x_start
            for n, c in enumerate(coefficients):
                self.profile[x + starting_index] += c * scaled_x**n
