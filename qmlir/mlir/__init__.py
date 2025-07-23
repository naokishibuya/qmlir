"""MLIR Integration for QMLIR

This module contains all MLIR-related functionality including
circuit transpilation to MLIR and optimization tools.
"""

from .transpiler import circuit_to_mlir, apply_passes


__all__ = ["circuit_to_mlir", "apply_passes"]
