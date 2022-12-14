__all__ = ["CamProtocol"]

from typing import Optional, Protocol
import numpy as np


class CamProtocol(Protocol):

    SIZE: int = ...
    profile: np.ndarray = ...

    def __call__(self, size: int, profile: Optional[np.ndarray]):
        ...

    def __repr__(self):
        ...
