import numpy as np


class Cam:
    def __init__(self, size: int = 7200, profile: np.ndarray = None):
        self.SIZE = size
        if profile is None:
            self.profile = np.ones((size))
        else:
            self.profile = self.fit_profile(profile)

    def fit_profile(self, profile: np.ndarray, degree: int = 60):
        if len(profile.shape) != 1:
            raise SyntaxError("Input profile dimension incorrect")

        if profile.shape[0] == self.SIZE:
            self.SIZE = profile
            return

        coefficients = np.polynomial.polynomial.polyfit(
            x=np.arange(start=0, stop=profile.shape[0], stop=1), y=profile, deg=degree
        )
