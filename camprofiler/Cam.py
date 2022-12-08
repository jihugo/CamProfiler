__all__ = ["Cam"]

import numpy as np
from typing import Optional
from camprofiler.protocol import CamProtocol
from camprofiler.utilities import circular_convolve
from stl import mesh
import sympy as sp


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
        self.profile = np.ones((size))
        if profile is not None:
            self.set_profile_with_straight_lines(profile)

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
        """Use rolling average to smoothen cam curve by circular convolution

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
            self.profile = circular_convolve(self.profile, kernel)

    def set_profile_with_function(
        self,
        function: sp.Expr,
        variable: sp.Symbol,
        function_start: float = 0,
        function_end: float = 1,
        cam_start: float = 0,
        cam_end: float = 360,
    ) -> None:
        """Use a function to set the cam profile

        Parameters
        ----------
        function : sp.Expr
            Function as SymPy expression

        variable : sp.Symbol
            The variable that defines the domain of the function

        function_start : float, default = 0
            The left bound of the domain of the function

        function_end : float, default = 1
            The right bound of the domain of the function

        cam_start : float, default = 0
            Angular value (0 - 360) that marks the start of the segment.
            Curve will be fitted to [start, end).

        cam_end : float, default = 360
            Angular value (0 - 360) that marks the end of the segment.
            Curve will be fitted to [start, end).
        """

        index = int(cam_start / 360 * self.SIZE + 0.5)
        ending_index = int(cam_end / 360 * self.SIZE + 0.5)
        n = ending_index - index
        t_arr = np.linspace(
            start=float(function_start), stop=float(function_end), num=n
        )
        for _, t in enumerate(t_arr):
            self.profile[index] = function.subs(variable, t)
            index += 1

    def get_2D(
        self,
        offset: float = 0.0,
        scale: float = 1.0,
    ) -> np.ndarray:
        """Get 2D geometry of cam

        Parameters
        ----------
        offset : float, default = 0.0
        scale : float = 1.0
            At each given degree, radius = offset + scale * profile

        Returns
        -------
        2D numpy array.
            First row contains x coordinates
            Second row contains y coordinates
        """

        twoD = np.ndarray((2, self.SIZE))
        for i, lift in enumerate(self.profile):
            r = offset + scale * lift
            theta = np.radians(360 * i / self.SIZE)
            twoD[0][i] = r * np.cos(theta)
            twoD[1][i] = r * np.sin(theta)
        return twoD

    def to_stl(
        self,
        file_name: str,
        offset: float = 0.0,
        scale: float = 1.0,
        thickness: float = 1,
    ) -> None:
        """Create stl file from cam profile

        Parameters
        ----------
        file_name : str
            Name of stl file that will be saved.
            Note: include ".stl" at the end.

        offset : float, default = 0.0
        scale : float, default = 1.0
            At each given degree, radius = offset + scale * profile

        thickness : float
            Thickness of cam stl
        """
        twoD = self.get_2D(offset, scale)
        solid = mesh.Mesh(np.zeros(((self.SIZE) * 6), dtype=mesh.Mesh.dtype))

        O0 = np.array([0, 0, 0])
        O1 = np.array([0, 0, thickness])

        p1 = np.array([twoD[0][0], twoD[1][0], 0])
        p3 = np.array([p1[0], p1[1], thickness])
        for i in range(self.SIZE - 1):
            p0 = p1
            p1 = np.array([twoD[0][i + 1], twoD[1][i + 1], 0])

            p2 = p3
            p3 = np.array([p1[0], p1[1], thickness])

            solid.vectors[6 * i] = np.array([O0, p0, p2])
            solid.vectors[6 * i + 1] = np.array([O0, O1, p2])
            solid.vectors[6 * i + 2] = np.array([O0, p0, p1])
            solid.vectors[6 * i + 3] = np.array([O1, p2, p3])
            solid.vectors[6 * i + 4] = np.array([p0, p1, p3])
            solid.vectors[6 * i + 5] = np.array([p0, p2, p3])

        # Stick the last 4 points:
        p0 = p1
        p2 = p3
        p1 = np.array([twoD[0][0], twoD[1][0], 0])
        p3 = np.array([p1[0], p1[1], thickness])

        i += 1
        solid.vectors[6 * i] = np.array([O0, p0, p2])
        solid.vectors[6 * i + 1] = np.array([O0, O1, p2])
        solid.vectors[6 * i + 2] = np.array([O0, p0, p1])
        solid.vectors[6 * i + 3] = np.array([O1, p2, p3])
        solid.vectors[6 * i + 4] = np.array([p0, p1, p3])
        solid.vectors[6 * i + 5] = np.array([p0, p2, p3])

        solid.save(file_name)
