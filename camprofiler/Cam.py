__all__ = ["Cam"]

import numpy as np
from typing import Optional
from camprofiler.protocol import CamProtocol
from camprofiler.utilities import seamless_convolve


class Cam(CamProtocol):
    """The Cam object

    Parameters
    ----------
    size : int, default = 36000
        Number of sample points in cam profile

    profile : np.ndarray, default = None
        Apply a cam profile.
        Flat profile is applied by default.
    """

    def __init__(self, size: int = 36000, profile: Optional[np.ndarray] = None):
        self.SIZE = size
        if profile is None:
            self.profile = np.ones((size))
        else:
            self.profile = self.set_with_straight_lines(profile)

    def set_profile_with_straight_lines(
        self,
        profile: np.ndarray,
        start: float = 0.0,
        end: float = 360.0,
    ):
        """Set a segment of the cam profile to straight lines defined by profile

        Parameters
        ----------
        profile : np.ndarray
            1D Numpy array or n x 2 Numpy array that define points.
            1D array represents evenly-spaced values
            n x 2 array contains angle in the first column and values in second
                column, these points must be in order with increasing angle.

        start : float, default = 0.0
            Angular value (0 - 360) that marks the start of the segment.
            Curve will be fitted to [start, end).

        end : float, default = 360.0
            Angular value (0 - 360) that marks the end of the segment.
            Curve will be fitted to [start, end).
        """

        if len(profile.shape) != 1:
            if profile.shape[1] != 2:
                raise SyntaxError("Input profile has incorrect format.")
            length = profile.shape[0]

        # convert 1D into 2D
        else:
            length = profile.shape[0]
            profile = np.array([np.arange(start=0, stop=length), profile]).transpose()

        # Apply 2D points
        starting_index = int(start / 360 * self.SIZE)
        ending_index = int(end / 360 * self.SIZE)

        idx1 = starting_index
        dummy_profile = np.concatenate([self.profile, [0]])
        starting_x = profile[0][0]
        ending_x = profile[-1][0]
        span = ending_x - starting_x
        for i in range(length - 1):
            curr_point = profile[i]
            next_point = profile[i + 1]

            idx0 = idx1
            idx1 = (
                starting_index
                + int(
                    (next_point[0] - starting_x)
                    / (span)
                    * (ending_index - starting_index)
                    + 0.5
                )
                + 1
            )

            dummy_profile[idx0:idx1] = np.linspace(
                start=curr_point[1], stop=next_point[1], num=int(idx1 - idx0)
            )
        self.profile = dummy_profile[:-1]

    def set_profile_polynomial_with_points(
        self,
        profile: np.ndarray,
        degree: int = 3,
        start: float = 0.0,
        end: float = 360.0,
    ):
        """Set a segment of the cam profile to a polynomial curve using
        points from the curve
        *Uses Numpy.polynomial.polynomial.polyfit

        Parameters
        ----------
        profile : np.ndarray
            1D Numpy array or n x 2 Numpy array that define points.
            1D array represents evenly-spaced values
            n x 2 array contains angle in the first column and values in second
                column. Only the segment from the first to the last point of the
                given profile will be fitted and applied.

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
            if profile.shape[1] != 2:
                raise SyntaxError("Input profile dimension incorrect.")
            # 2D array
            coefficients = np.polynomial.polynomial.polyfit(
                x=profile[:, 0], y=profile[:, 1], deg=degree
            )
            self.set_profile_polynomial_with_coefficients(
                coefficients,
                cam_start=start,
                cam_end=end,
                x_start=profile[0][0],
                x_end=profile[-1][0],
            )

        # 1D array
        else:
            coefficients = np.polynomial.polynomial.polyfit(
                x=np.linspace(start=0, stop=1, num=profile.shape[0]),
                y=profile,
                deg=degree,
            )
            self.set_profile_polynomial_with_coefficients(
                coefficients, cam_start=start, cam_end=end
            )

    def set_profile_polynomial_with_coefficients(
        self,
        coefficients: np.ndarray,
        cam_start: float = 0.0,
        cam_end: float = 360.0,
        x_start: int = 0,
        x_end: int = 1,
    ):
        """Set a segment of the cam profile to a polynomial curve using coefficients

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

    def rolling_average_smoothen(
        self, kernel_size_in_degrees: int = 3, num_iterations: int = 2
    ):
        """Use rolling average to smoothen cam curve by seamless convolution

        Parameters
        ----------
        kernel_size_in_degrees : int, default = 3
            Kernel size in degrees of cam rotation

        num_iterations : int, default = 2
            Number of times the rolling average is applied
        """
        kernel_size = int(kernel_size_in_degrees / 360 * self.SIZE)
        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = np.ones(kernel_size) / kernel_size
        for i in range(num_iterations):
            self.profile = seamless_convolve(self.profile, kernel)
