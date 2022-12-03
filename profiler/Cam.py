import numpy as np


class Cam:
    def __init__(self, size: int = 36000, profile: np.ndarray = None):
        self.SIZE = size
        if profile is None:
            self.profile = np.ones((size))
        else:
            self.profile = self.fit_profile(profile)

    def fit_profile_polynomial(
        self,
        profile: np.ndarray,
        degree: int = 10,
        start: float = 0.0,
        end: float = 360.0,
    ):
        if len(profile.shape) != 1:
            raise SyntaxError("Input profile dimension incorrect")

        starting_index = int(start / 360 * self.SIZE)
        ending_index = int(end / 360 * self.SIZE)

        if profile.shape[0] == ending_index - starting_index:
            self.profile[starting_index:ending_index] = profile
            return

        coefficients = np.polynomial.polynomial.polyfit(
            x=np.linspace(start=0, stop=1, num=profile.shape[0]),
            y=profile,
            deg=degree,
        )

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
