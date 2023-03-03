"""Cam Protocol"""
__all__ = ["CamProtocol"]

from typing import Protocol, Iterable


class CamProtocol(Protocol):
    """Cam Protocol class"""

    index_per_degree: int = ...
    profile: Iterable[float] = ...
