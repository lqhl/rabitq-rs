"""
Python bindings for RaBitQ-RS.

This package provides high-performance approximate nearest neighbor search
using both MSTG (Multi-Scale Tree Graph) and IVF (Inverted File) algorithms
with RaBitQ quantization.
"""

from .rabitq_rs import IvfRabitqIndex, MstgIndex

__version__ = "0.5.0"
__all__ = ["IvfRabitqIndex", "MstgIndex"]
