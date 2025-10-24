"""
Python bindings for RaBitQ-RS MSTG index.

This package provides high-performance approximate nearest neighbor search
using the MSTG (Multi-Scale Tree Graph) algorithm with RaBitQ quantization.
"""

from .rabitq_rs import MstgIndex

__version__ = "0.4.0"
__all__ = ["MstgIndex"]
