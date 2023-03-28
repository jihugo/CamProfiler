"""Cam Protocol"""
__all__ = ["CamProtocol"]

from typing import Protocol, Iterable


class CamProtocol(Protocol):
    """Cam Protocol class"""

    profile: Iterable[float] = ...
    SIZE: int = ...
