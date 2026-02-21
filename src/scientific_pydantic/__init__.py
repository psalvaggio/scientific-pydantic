"""Pydantic adapters for common scientific libraries

Adapters in the root of this package are for Python standard library types ONLY.

Subpackages shall follow the structure of their library and exist at the same
path as the type they are adapting. For instance, the adapter
`astropy.units.UnitBase` lives in `scientific_pydantic.astropy.units`.
"""

from . import astropy, numpy, scipy, shapely
from .range import RangeAdapter
from .slice import IntSliceAdapter, SliceAdapter

__all__ = [
    "IntSliceAdapter",
    "RangeAdapter",
    "SliceAdapter",
    "astropy",
    "numpy",
    "scipy",
    "shapely",
]
