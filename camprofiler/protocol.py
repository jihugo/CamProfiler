"""Cam Protocol"""
__all__ = ["CamProtocol"]

from typing import Protocol, Iterable


class CamProtocol(Protocol):
    """Cam Protocol class"""

    profile: Iterable[float] = ...
    SIZE: int = ...

    def get_2d(self):
        """Get 2D cartesian coordinates of cam"""

    def get_stl(self):
        """Get STL file of cam"""
