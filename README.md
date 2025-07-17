# Quantum MLIR Dialect

A quantum computing dialect experiment for MLIR with gate cancellation optimization.

## Features

* **Quantum Operations**: `quantum.alloc`, `quantum.x`, `quantum.h`, `quantum.cx`
* **Optimization Pass**: Cancels consecutive X gates (`X; X` → identity)
* **Standalone Build**: External dialect, no LLVM source modifications needed
* **Code Formatting**: Automated Python (ruff) and C++ (clang-format) formatting with pre-commit hooks

## Quick Start

**1. Clone the Repository:**

```bash
git clone https://github.com/naokishibuya/quantum-compiler.git
cd quantum-compiler
```

**2. Setup Python Environment:**

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install pytest ruff pre_commit
pre-commit install
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
cd ../../quantum-compiler
mkdir build
cd build
cmake -G Ninja ..
ninja
```

**6. Install QMLIR Python Package (Development):**

For development, install the QMLIR package in editable mode:

```bash
cd ..  # quantum-compiler root
pip install -e .
```

This makes the `qmlir` package available in your venv. The package automatically finds:
- MLIR Python bindings from the LLVM build directory
- The `quantum-opt` executable from the quantum dialect build

**Note**: The `qmlir/config.py` module handles automatic discovery of MLIR dependencies, so you don't need to set up paths manually.

**7. Verification:**

At the project root, run:

```bash
echo 'module {
  func.func @test() {
    %q = "quantum.alloc"() : () -> i32
    "quantum.x"(%q) : (i32) -> ()
    "quantum.x"(%q) : (i32) -> ()
    "quantum.h"(%q) : (i32) -> ()
    return
  }
}' | ./build/mlir/tools/quantum-opt --quantum-cancel-x
```

Output (X gates cancelled):
```mlir
module {
  func.func @test() {
    %0 = "quantum.alloc"() : () -> i32
    "quantum.h"(%0) : (i32) -> ()
    return
  }
}
```

Test the Python library:

```bash
python -c 'from qmlir import Circuit; c = Circuit(); c.h(0).cx(0,1); print("QMLIR working!")'
```

## Testing

**1. Run MLIR tests:**

At the project root, run:

```bash
./build/mlir/tools/quantum-opt mlir/test/Dialect/Quantum/double_x_test.mlir --quantum-cancel-x
./build/mlir/tools/quantum-opt mlir/test/Dialect/Quantum/bell_state.mlir --verify-diagnostics
```

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
- Add gate operations in `qmlir/ast.py`
- Update MLIR generation in `qmlir/codegen.py`
- Add tests in `tests/`

**4. Development Workflow:**
- All changes are automatically reflected due to `pip install -e .`
- Run tests frequently with `pytest`
- Use `examples/` for interactive development
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

## Architecture

The quantum compiler follows a layered architecture:

1. **MLIR Quantum Dialect** (`mlir/`): Core quantum operations and passes
2. **Python Frontend** (`qmlir/`): High-level circuit construction API
3. **Integration Layer** (`tests/`, `examples/`): Testing and demonstrations
4. **Build System** (`CMakeLists.txt`, `pyproject.toml`): Development and packaging
