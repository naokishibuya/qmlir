"""Configuration for QMLIR library and tests.

This module handles path setup and configuration for the QMLIR library.
"""

import os
import sys
from pathlib import Path


def setup_mlir_path():
    """Set up the MLIR Python bindings path."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent

    # Try multiple possible locations for MLIR Python bindings
    possible_paths = [
        # From LLVM build directory (primary location after ninja MLIRPythonModules)
        project_root / ".." / "llvm-project" / "build" / "tools" / "mlir" / "python_packages" / "mlir_core",
        # From our build directory (copied from LLVM install - legacy)
        project_root / "build" / "python_packages" / "mlir_core",
        # From LLVM install directory (if someone did ninja install)
        project_root / ".." / "llvm-project" / "install" / "python_packages" / "mlir_core",
        # From environment variable
        Path(os.environ.get("MLIR_PYTHON_PACKAGES", "")) / "mlir_core"
        if os.environ.get("MLIR_PYTHON_PACKAGES")
        else None,
        # From external LLVM build (if separate)
        project_root / "llvm-project" / "build" / "tools" / "mlir" / "python_packages" / "mlir_core",
        # Legacy fallback to venv
        project_root / "venv" / "python_packages" / "mlir_core",
    ]

    # Filter out None values
    possible_paths = [path for path in possible_paths if path is not None]

    for mlir_path in possible_paths:
        if mlir_path.exists():
            sys.path.insert(0, str(mlir_path))
            return

    raise RuntimeError(f"MLIR Python bindings not found in any of: {possible_paths}")


def get_quantum_opt_path():
    """Get the path to the quantum-opt executable."""
    project_root = Path(__file__).parent.parent
    quantum_opt = project_root / "build" / "mlir" / "tools" / "quantum-opt"

    if quantum_opt.exists():
        return str(quantum_opt)
    else:
        raise RuntimeError(f"quantum-opt not found at: {quantum_opt}")


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


# Initialize MLIR path when this module is imported
setup_mlir_path()
