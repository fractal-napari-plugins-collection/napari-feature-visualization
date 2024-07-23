try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"
from ._sample_data import make_sample_data
from .feature_vis import (
    feature_vis,
)

__all__ = (
    "make_sample_data",
    "feature_vis",
)
