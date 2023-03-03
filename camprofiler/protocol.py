"""Cam Protocol"""
__all__ = ["CamProtocol"]

from typing import Optional, Protocol
import numpy as np


class CamProtocol(Protocol):
    """Cam Protocol class"""

    index_per_degree: int = ...
    profile: np.ndarray = ...
