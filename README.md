# Quantum MLIR Dialect

A quantum computing dialect experiment for MLIR with gate cancellation optimization.

## Features

* **Quantum Operations**: `quantum.alloc`, `quantum.x`, `quantum.h`, `quantum.cx`
* **Optimization Pass**: Cancels consecutive X gates (`X; X` → identity)
* **Standalone Build**: External dialect, no LLVM source modifications needed

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

**3. Build LLVM/MLIR:**

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

Build MLIR Python bindings (but don't install them to venv):

```bash
ninja MLIRPythonModules
```

**Important**: Do NOT run `ninja install` or `ninja install-MLIRPythonModules` as this will copy many files to your venv, which is not needed for development. The quantum compiler will find the MLIR Python bindings automatically from the LLVM build directory at `build/tools/mlir/python_packages/`.

**4. Build Quantum Dialect:**

Go back to the quantum compiler project root and build the quantum dialect:

```bash
cd ../../quantum-compiler
mkdir build
cd build
cmake -G Ninja ..
ninja
```

**5. Install QMLIR Python Package (Development):**

For development, install the QMLIR package in editable mode:

```bash
cd ..  # quantum-compiler root
pip install -e .
```

This makes the `qmlir` package available in your venv. The package automatically finds:
- MLIR Python bindings from the LLVM build directory
- The `quantum-opt` executable from the quantum dialect build

**Note**: The `qmlir/config.py` module handles automatic discovery of MLIR dependencies, so you don't need to set up paths manually.

**6. Verification:**

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
python -c "from qmlir import Circuit; c = Circuit(); c.h(0).cx(0,1); print('QMLIR working!')"
```

## Installation for Users

For users who want to install the quantum compiler:

```bash
# Install from source
git clone https://github.com/naokishibuya/quantum-compiler.git
cd quantum-compiler
pip install .
```

Or:

```bash
# Install directly from GitHub
pip install git+https://github.com/naokishibuya/quantum-compiler.git
```

This installs:
- `qmlir` Python package
- `qmlir-opt` command-line tool (optional)

**Note**: Users need to have LLVM/MLIR with Python bindings installed separately.

## Testing

**1. Run MLIR tests:**

At the project root, run:

```bash
./build/mlir/tools/quantum-opt mlir/test/Dialect/Quantum/double_x_test.mlir --quantum-cancel-x
./build/mlir/tools/quantum-opt mlir/test/Dialect/Quantum/bell_state.mlir --verify-diagnostics
```

**2. QMLIR Python Library:**

The quantum compiler includes QMLIR, a Python library for building quantum circuits:

```bash
# Set up the environment
source venv/bin/activate

# Run demonstrations
python examples/demo_bell_state.py
python examples/demo_double_x.py

# Run unit tests (from project root)
python -m pytest tests/ -v

# Pipe to quantum-opt for processing
python examples/demo_bell_state.py | build/mlir/tools/quantum-opt
python examples/demo_double_x.py | build/mlir/tools/quantum-opt --quantum-cancel-x
```

Note: The QMLIR library automatically finds MLIR Python bindings in:
1. `../llvm-project/build/tools/mlir/python_packages/mlir_core` (primary location after LLVM build)
2. `build/python_packages/mlir_core` (fallback for older build layouts)
3. `$MLIR_PYTHON_PACKAGES/mlir_core` (if environment variable is set)

## Examples

### Simple Circuit

```python
from qmlir import Circuit

circuit = Circuit()
circuit.h(0)  # Hadamard gate on qubit 0
circuit.cx(0, 1)  # CNOT gate
circuit.measure(0, 0)  # Measure qubit 0 into classical bit 0

print(circuit.to_mlir())
```

### Bell State Generation

```python
from qmlir import Circuit

# Create a Bell state |00⟩ + |11⟩
circuit = Circuit()
circuit.h(0)
circuit.cx(0, 1)

# Generate MLIR and optimize
mlir_code = circuit.to_mlir()
optimized = circuit.optimize(mlir_code)
print(optimized)
```

See `examples/` directory for more complete examples.

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test files
pytest tests/test_circuit.py -v
pytest tests/test_mlir.py -v
pytest tests/test_integration.py -v
```

This will run 21 unit tests covering:
- Circuit construction and gate operations
- MLIR code generation and formatting
- Quantum pass integration with quantum-opt

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
- Run tests frequently with `pytest tests/`
- Use `examples/` for interactive development
- Test integration with `quantum-opt` tool

## Architecture

The quantum compiler follows a layered architecture:

1. **MLIR Quantum Dialect** (`mlir/`): Core quantum operations and passes
2. **Python Frontend** (`qmlir/`): High-level circuit construction API
3. **Integration Layer** (`tests/`, `examples/`): Testing and demonstrations
4. **Build System** (`CMakeLists.txt`, `pyproject.toml`): Development and packaging

This separation allows for:
- Independent development of MLIR dialect and Python frontend
- Easy testing and validation at each layer
- Professional packaging and distribution
- Extension to other frontends (C++, Rust, etc.)
