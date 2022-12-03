__all__ = ["Analyzer"]

import numpy as np
import camprofiler.Cam


class Analyzer:
    def __init__(self, cam: camprofiler.Cam):
        self.cam = cam
