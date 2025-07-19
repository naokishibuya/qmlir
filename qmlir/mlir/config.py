"""MLIR Configuration and Setup

This module handles MLIR configuration and setup for the QMLIR library.
"""

import os
import sys
from pathlib import Path


def ensure_llvm():
    """Ensure LLVM/MLIR Python bindings are available in the Python path.

    This function checks for MLIR Python bindings and adds them to sys.path
    if they're not already there. It looks for bindings in:
    1. MLIR_PYTHON_PACKAGES environment variable
    2. Default LLVM build directory

    Returns:
        str: Path to the MLIR Python bindings directory

    Raises:
        RuntimeError: If MLIR Python bindings are not found
    """
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent

    # Check environment variable first
    if os.environ.get("MLIR_PYTHON_PACKAGES"):
        mlir_path = Path(os.environ["MLIR_PYTHON_PACKAGES"]) / "mlir_core"
        if mlir_path.exists():
            # Only add to path if not already there
            if str(mlir_path) not in sys.path:
                sys.path.insert(0, str(mlir_path))
            return str(mlir_path)

    # Default to LLVM build directory
    mlir_path = project_root / ".." / "llvm-project" / "build" / "tools" / "mlir" / "python_packages" / "mlir_core"
    if mlir_path.exists():
        # Only add to path if not already there
        if str(mlir_path) not in sys.path:
            sys.path.insert(0, str(mlir_path))
        return str(mlir_path)

    raise RuntimeError(
        f"MLIR Python bindings not found. Tried:\n1. MLIR_PYTHON_PACKAGES environment variable\n2. {mlir_path}"
    )


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


# Ensure LLVM/MLIR bindings are available when this module is imported
ensure_llvm()
