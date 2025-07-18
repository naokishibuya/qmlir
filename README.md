# QMLIR - Quantum Circuit Compiler for MLIR

This project implements a quantum computing dialect for MLIR (Multi-Level Intermediate Representation). It allows for the construction and optimization of quantum circuits using a high-level Python API, while leveraging the power of MLIR for backend compilation.

QMLIR is an experimental project that supports only a subset of quantum operations, focusing on basic gates and their optimizations. The main feature is the cancellation of consecutive self-inverse gates, which simplifies quantum circuits by removing redundant operations.

## Features

* **Quantum Operations**: `quantum.alloc`, `quantum.i`, `quantum.x`, `quantum.y`, `quantum.z`, `quantum.h`, `quantum.cx`
* **Optimization Pass**: Cancels consecutive self-inverse gates (`I; I` → `I`, `X; X` → identity, `Y; Y` → identity, `Z; Z` → identity, `H; H` → identity, `CX; CX` → identity)
* **Simple API**: Just three imports needed: `QuantumCircuit`, `Parameter`, `simulate`
* **JAX Runtime**: High-performance quantum circuit simulation with JIT compilation and GPU acceleration
* **Standalone Build**: External dialect, no LLVM source modifications needed
* **Code Formatting**: Automated Python (ruff) and C++ (clang-format) formatting with pre-commit hooks

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
}' | ./build/mlir/tools/quantum-opt --quantum-cancel-self-inverse
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
python -c 'from qmlir import QuantumCircuit, simulate; c = QuantumCircuit(2); c.h(0).cx(0,1); results = simulate(c); print("Bell state probabilities:", results["probabilities"])'
```

**Minimal API Example:**

```bash
python examples/bell_state_demo.py
```

**Interactive Learning:**

```bash
cd notebooks
jupyter notebook
```

**Full Documentation:**

```bash
python examples/optimization_demo.py
python examples/parameterized_demo.py
```

The examples demonstrate Bell states, circuit optimization, and parameterized circuits using QMLIR's simple 3-function API. The notebooks provide interactive tutorials with detailed explanations.
```
QMLIR + JAX Quantum Simulation Example
=============================================
1. Creating Bell state circuit...
   Circuit: H(0) → CX(0,1)
   Gates: 2

2. Simulating circuit...
   → Compiling to MLIR
   → Optimizing with quantum-opt
   → Simulating with JAX backend
   ✓ Simulation complete

3. Results:
   Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
   -----------------------------------
   |00⟩:  0.7071 → P = 0.5000
   |01⟩:  0.0000 → P = 0.0000
   |10⟩:  0.0000 → P = 0.0000
   |11⟩:  0.7071 → P = 0.5000

4. Verification:
   Expected P(|00⟩) = P(|11⟩) = 0.5
   Actual   P(|00⟩) = 0.5000
   Actual   P(|11⟩) = 0.5000
   Bell state created: ✓ YES

=============================================
JAX simulation working correctly! 🎉
```

## Testing

**1. Run MLIR tests:**

At the project root, run:

```bash
./build/mlir/tools/quantum-opt mlir/test/Dialect/Quantum/double_x_test.mlir --quantum-cancel-self-inverse
./build/mlir/tools/quantum-opt mlir/test/Dialect/Quantum/bell_state.mlir --verify-diagnostics
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
- Add gate operations in `qmlir/circuit.py`
- Update MLIR generation in `qmlir/mlir_generator.py`
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
3. **Simulation Interface** (`qmlir/simulator.py`): High-level simulation functions
4. **JAX Runtime** (`qmlir/runtime/`): High-performance JAX-based simulation backend
5. **Integration Layer** (`tests/`, `examples/`): Testing and demonstrations
6. **Build System** (`CMakeLists.txt`, `pyproject.toml`): Development and packaging
