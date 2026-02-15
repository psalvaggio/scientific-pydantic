"""Pydantic adapters for common scientific libraries"""

from . import numpy, shapely
from .range import RangeAdapter
from .slice import IntSliceAdapter, SliceAdapter

__all__ = [
    "IntSliceAdapter",
    "RangeAdapter",
    "SliceAdapter",
    "numpy",
    "shapely",
]
