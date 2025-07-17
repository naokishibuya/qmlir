# Quantum MLIR Dialect

A standalone quantum computing dialect for MLIR with gate cancellation optimization.

## Features

* **Quantum Operations**: `quantum.alloc`, `quantum.x`, `quantum.h`, `quantum.cx`
* **Optimization Pass**: Cancels consecutive X gates (`X; X` → identity)
* **Standalone Build**: External dialect, no LLVM source modifications needed

## Quick Start

**1. Setup Python Environment:**

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

**1. Build LLVM/MLIR:**

```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
```

Install Python requirements for MLIR:

```bash
pip install -r mlir/python/requirements.txt
```

Build LLVM and MLIR:

```bash
mkdir build
cd build
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DCMAKE_INSTALL_PREFIX="$VIRTUAL_ENV"
ninja
```

Install MLIR Python bindings:

```bash
ninja install-MLIRPythonModules
```

**2. Build Quantum Dialect:**

```bash
git clone https://github.com/naokishibuya/quantum-compiler.git
cd quantum-compiler
mkdir build
cd build
cmake -G Ninja ..
ninja
```

**3. Verification:**

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

## Testing

**Run MLIR tests:**

At the project root, run:

```bash
./build/mlir/tools/quantum-opt mlir/test/Dialect/Quantum/double_x_test.mlir --quantum-cancel-x
./build/mlir/tools/quantum-opt mlir/test/Dialect/Quantum/bell_state.mlir --verify-diagnostics
```

**Python bindings:**

At the project root, run:

```bash
# Make sure PYTHONPATH includes the MLIR modules
export PYTHONPATH=$(pwd)/venv/python_packages/mlir_core
python mlir/python/mlir/dialects/quantum/emit_quantum_mlir.py build/mlir/tools/quantum-opt 
```

## Development

**Add Operations:**

- Edit `QuantumOps.td`
- Implement in `QuantumOps.cpp`
- Add tests

**Add Passes:**
- Edit `Passes.td`
- Implement in `Passes/`
- Register in `QuantumPasses.h`
