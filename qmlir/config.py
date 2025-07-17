"""Configuration for QMLIR library and tests.

This module handles path setup and configuration for the QMLIR library.
"""

import os
import sys
import subprocess
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
    """Get the path to the quantum-opt executable.

    Returns:
        str: Path to quantum-opt executable.

    Raises:
        RuntimeError: If quantum-opt is not found.
    """
    project_root = Path(__file__).parent.parent
    quantum_opt = project_root / "build" / "mlir" / "tools" / "quantum-opt"

    if quantum_opt.exists():
        return str(quantum_opt)
    else:
        raise RuntimeError(f"quantum-opt not found at: {quantum_opt}")


def run_quantum_opt(mlir_code, *args, timeout=10):
    """Run quantum-opt with the given MLIR code and arguments.

    Args:
        mlir_code (str): The MLIR code to process.
        *args: Additional command-line arguments to pass to quantum-opt.
        timeout (int): Timeout in seconds for the subprocess call.

    Returns:
        subprocess.CompletedProcess: The result of the subprocess call.

    Raises:
        RuntimeError: If quantum-opt is not found.
    """
    quantum_opt_path = get_quantum_opt_path()
    command = [quantum_opt_path] + list(args)

    return subprocess.run(command, input=mlir_code, capture_output=True, text=True, timeout=timeout)


# Initialize MLIR path when this module is imported
setup_mlir_path()
