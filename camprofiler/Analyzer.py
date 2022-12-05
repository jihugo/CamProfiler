__all__ = ["Analyzer"]

import numpy as np
from camprofiler.protocol import CamProtocol
from camprofiler.utilities import seamless_convolve
from typing import *


class Analyzer:
    """The cam analyzer object

    Attributes
    ----------
    stats : Dict
        Dictionary of statistics
    """

    def __init__(self):
        self.stats: Dict = {}

    def analyze(self, cam: CamProtocol):
        """Perform analysis on cam

        Parameters
        ----------
        cam: CamProtocol
            Perform analysis on this cam.
        """
        self.get_PVAJ(cam)

    def get_PVAJ(self, cam: CamProtocol, **kwargs):
        """Calculate position, velocity, acceleration, and jerk using convolution
        Note: higher definition cam profile results in more accurate results.

        Parameters
        ----------
        cam : CamProtocol
            Perform calculations on this cam

        stride : float, default = 0.1

        """
        stride = kwargs["stride"] if "stride" in kwargs else 0.1
        deg_per_sample = 360.0 / cam.SIZE
        k = int(stride / deg_per_sample)
        k = np.min([k, 3])

        diff_kernel = np.zeros(k)
        diff_kernel[-1] = -1 / (deg_per_sample * (k - 1))
        diff_kernel[0] = 1 / (deg_per_sample * (k - 1))

        ave_kernel = np.ones(k) / k

        self.stats["position"] = cam.profile

        velocity = seamless_convolve(cam.profile, diff_kernel)
        velocity = seamless_convolve(velocity, ave_kernel)
        self.stats["velocity"] = velocity

        accel = seamless_convolve(velocity, diff_kernel)
        accel = seamless_convolve(accel, ave_kernel)
        self.stats["acceleration"] = accel

        jerk = seamless_convolve(accel, diff_kernel)
        jerk = seamless_convolve(jerk, ave_kernel)
        self.stats["jerk"] = jerk
