# QMLIR - Quantum Circuit Compiler for MLIR

This project implements a quantum computing dialect for MLIR (Multi-Level Intermediate Representation). It allows for the construction and optimization of quantum circuits using a high-level Python API, while leveraging the power of MLIR for backend compilation.

QMLIR is an experimental project that supports only a subset of quantum operations, focusing on basic gates and their optimizations. The main feature is the cancellation of consecutive self-inverse gates, which simplifies quantum circuits by removing redundant operations.

## Quick Start

**1. Clone the Repository:**

```bash
git clone https://github.com/naokishibuya/qmlir.git
cd qmlir
```

**2. Setup Python Environment:**

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pre-commit install
pre-commit install --hook-type pre-push
```

**3. Install clang-format (for C++ code formatting):**

**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt install clang-format
```

**macOS:**
```bash
# Using Homebrew
brew install clang-format

# Or using MacPorts
sudo port install clang-format
```

**4. Build LLVM/MLIR:**

At the project sibling directory, clone llvm-project:

```bash
cd ..
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
```

Install Python requirements for MLIR:

```bash
pip install -r mlir/python/requirements.txt
```

Build LLVM and MLIR for development:

```bash
mkdir build
cd build
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DPython3_EXECUTABLE=$(which python)
ninja
```

**5. Build Quantum Dialect:**

Go back to the quantum compiler project root and build the quantum dialect:

```bash
cd ../../qmlir
mkdir build
cd build
cmake -G Ninja ..
ninja
```

**6. Install QMLIR Python Package (Development):**

For development, install the QMLIR package in editable mode:

```bash
cd ..  # qmlir root
pip install -e .
```

This makes the `qmlir` package available in your venv, automatically discovering MLIR Python bindings from `../llvm-project/build/tools/mlir/python_packages/mlir_core` (or `MLIR_PYTHON_PACKAGES` environment variable) and the `quantum-opt` executable from the local build.

## Testing

**1. Run MLIR tests:**

At the build directory, you can run the MLIR tests for the Quantum dialect:

```bash
cd ./build
ninja check-quantum
```

Or munually run the `quantum-opt` tool on test files (.mlir) using:

```bash
./build/mlir/tools/quantum-opt mlir/test/Dialect/Quantum/double_x_test.mlir
```

With optimizations for self-inverse gates:

```bash
./build/mlir/tools/quantum-opt mlir/test/Dialect/Quantum/double_x_test.mlir --quantum-cancel-self-inverse
```

There are more examples in `examples/` directory.

**2. QMLIR Python Library:**

Run unit tests (from project root)

```bash
pytest
```

See `examples/` directory for Python examples.

## Development

**1. Add Operations:**

- Edit `mlir/include/mlir/Dialect/Quantum/IR/QuantumOps.td`
- Implement in `mlir/lib/Dialect/Quantum/QuantumOps.cpp`
- Add tests in `mlir/test/Dialect/Quantum/`

**2. Add Passes:**
- Edit `mlir/include/mlir/Dialect/Quantum/Passes/Passes.td`
- Implement in `mlir/lib/Dialect/Quantum/Passes/`
- Register in `mlir/include/mlir/Dialect/Quantum/Passes/Passes.h`

**3. Extend Python Library:**
- Add gate operations in `qmlir/circuit.py`, etc.
- Update MLIR generation in `qmlir/mlir/transpiler.py`
- Add tests in `tests/`

**4. Development Workflow:**
- All changes are automatically reflected due to `pip install -e .`
- Run tests frequently with `pytest`
- Use `notebooks/` for interactive development
- Test integration with `quantum-opt` tool

**5. Code Formatting:**

The project uses automated code formatting:

**Python code** (via ruff):
```bash
# Format Python files
ruff format .

# Run linting
ruff check .
```

**C++ code** (via clang-format):
```bash
# Format all C++ files
./format-cpp.sh

# Format a single file
clang-format -i mlir/lib/Dialect/Quantum/QuantumDialect.cpp

# Check formatting without changing files
clang-format --dry-run --Werror <file>
```

**Pre-commit hooks** (runs automatically on commit):
```bash
# Run all hooks on all files
pre-commit run --all-files

# Run specific hook
pre-commit run clang-format --all-files
pre-commit run ruff --all-files
```
