"""Module containing Cam object"""
__all__ = ["Cam"]

import numpy as np
from typing import Optional
from camprofiler.protocol import CamProtocol
from camprofiler.utilities import circular_convolve, circular_linear_interp
from stl import mesh
import sympy as sp


class Cam(CamProtocol):
    """The Cam object

    Attributes
    ----------
    SIZE : int
        Number of sample points in cam profile
    profile : NumPy ndarray | None
        1D NumPy array with length SIZE representing cam profile


    Parameters
    ----------
    size : int, default = 36000
        Number of sample points in cam profile

    profile : np.ndarray, default = None
        Flat profile of 1 is applied by default.
        Note: When profile is specified, SIZE is set to the size
        of provided profile instead of previous argument.
    """

    SIZE: int
    profile: Optional[np.ndarray]

    def __init__(self, size: int = 36000, profile: Optional[np.ndarray] = None):
        CamProtocol.__init__(self)
        if profile is not None:
            self.SIZE = np.shape(profile)[0]
            self.profile = profile
        else:
            self.SIZE = size
            self.profile = np.ones((size))

    def resize(self, size: int) -> None:
        """Linearly resize cam profile with specified size

        This calls the circular linear interp function in utilities module.

        Parameter
        ---------
        size : int
            New size of cam
        """
        return circular_linear_interp(self.profile, size)

    def set_profile(
        self,
        profile: np.ndarray,
        cam_start: float = 0.0,
        cam_end: float = 360.0,
        circular: Optional[bool] = None,
    ) -> None:
        """Set cam profile

        Parameters
        ----------
        profile : 1D NumPy array
            New profile to be applied

        cam_start : float, default = 0.0
            Angular value (0 - 360) that marks the start of the segment.
            Curve will be fitted to [start, end).

        cam_end : float, default = 360.0
            Angular value (0 - 360) that marks the end of the segment.
            Curve will be fitted to [start, end).

        circular : bool | None, default = None
            Provided profile is circular
            If profile is circular and needs to be resized, the circular
            linear interpolator is used to minimize discontinuity
        """
        if len(profile.shape) != 1:
            raise SyntaxError("Input profile is not 1D NumPy array")

        if circular is None:
            circular = cam_start == 0.0 and cam_end == 360.0

        starting_index = int(cam_start / 360 * self.SIZE)
        ending_index = int(cam_end / 360 * self.SIZE)
        span = ending_index - starting_index

        # Need to resize
        if span != profile.shape[0]:
            if circular:
                profile = circular_linear_interp(profile, span)
            else:
                profile = np.interp(
                    np.linspace(0, profile.shape[0], span),
                    np.arange(profile.shape[0]),
                    profile,
                )

        self.profile[starting_index:ending_index] = profile
