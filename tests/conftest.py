import os
import sys


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


# Add the QMLIR library to the path (now at root level)
qmlir_path = os.path.join(CURRENT_DIR, "..")
sys.path.insert(0, qmlir_path)


# Add the MLIR Python bindings to the path
# This assumes the virtual environment is set up correctly
# and the MLIR Python bindings are installed in the expected location.
venv_path = os.path.join(CURRENT_DIR, "..", "venv")
mlir_path = os.path.join(venv_path, "python_packages", "mlir_core")
sys.path.insert(0, mlir_path)
