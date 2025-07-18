"""MLIR Configuration and Setup

This module handles MLIR configuration and setup for the QMLIR library.
"""

import os
import sys
from pathlib import Path


def setup_mlir_path():
    """Set up the MLIR Python bindings path."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent

    # Check environment variable first
    if os.environ.get("MLIR_PYTHON_PACKAGES"):
        mlir_path = Path(os.environ["MLIR_PYTHON_PACKAGES"]) / "mlir_core"
        if mlir_path.exists():
            sys.path.insert(0, str(mlir_path))
            return

    # Default to LLVM build directory
    mlir_path = project_root / ".." / "llvm-project" / "build" / "tools" / "mlir" / "python_packages" / "mlir_core"
    if mlir_path.exists():
        sys.path.insert(0, str(mlir_path))
        return

    raise RuntimeError(f"MLIR Python bindings not found at: {mlir_path}")


def get_quantum_opt_path():
    """Get the path to the quantum-opt executable.

    Returns:
        str: Path to quantum-opt executable.

    Raises:
        RuntimeError: If quantum-opt is not found.
    """
    project_root = Path(__file__).parent.parent.parent
    quantum_opt = project_root / "build" / "mlir" / "tools" / "quantum-opt"

    if quantum_opt.exists():
        return str(quantum_opt)
    else:
        raise RuntimeError(f"quantum-opt not found at: {quantum_opt}")


# Initialize MLIR path when this module is imported
setup_mlir_path()
