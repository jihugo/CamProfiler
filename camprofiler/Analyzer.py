__all__ = ["Analyzer"]

import numpy as np
from camprofiler.protocol import CamProtocol
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

    def get_PVAJ(self, cam: CamProtocol):
        """Calculate postion, velocity, acceleration, and jerk

        Parameters
        ----------
        cam : CamProtocol
            Perform calculations on this cam.
        """
        self.stats["position"] = cam.profile

        degree_per_sample = 360 / cam.SIZE

        velocity = np.zeros((cam.SIZE - 1))
        acceleration = np.zeros((cam.SIZE - 2))
        jerk = np.zeros((cam.SIZE - 3))

        for i in range(cam.SIZE - 1):
            velocity[i] = (cam.profile[i + 1] - cam.profile[i]) / degree_per_sample
        self.stats["velocity"] = velocity

        for i in range(cam.SIZE - 2):
            acceleration[i] = (velocity[i + 1] - velocity[i]) / degree_per_sample
        self.stats["acceleration"] = acceleration

        for i in range(cam.SIZE - 3):
            jerk[i] = (acceleration[i + 1] - acceleration[i]) / degree_per_sample
        self.stats["jerk"] = jerk
