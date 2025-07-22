import os
import lit.formats


# ruff: noqa: F821
config.name = "QMLIR Quantum dialect tests"
config.test_format = lit.formats.ShTest()
config.suffixes = [".mlir"]

# Set the path to the LLVM and MLIR binaries assuming LLVM is at sibling directory of qmlir
CURRENT_DIR = os.path.dirname(__file__)
LLVM_BINARY_DIR = os.path.join(CURRENT_DIR, "..", "..", "..", "llvm-project", "build", "bin")
MLIR_BINARY_DIR = os.path.join(CURRENT_DIR, "..", "..", "build", "mlir", "tools")
config.environment["PATH"] = os.pathsep.join(
    [LLVM_BINARY_DIR, MLIR_BINARY_DIR] + config.environment.get("PATH", "").split(os.pathsep)
)
