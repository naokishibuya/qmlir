# Quantum MLIR Dialect and Gate Cancellation Pass

## Project Description

This repository contains the source files for a **minimal Quantum Dialect** in MLIR and an experimental **C++ optimization pass** that cancels redundant gate sequences.

It demonstrates key compiler concepts for quantum IR:

* **IR Design:** Defining custom quantum gate operations
* **Pass Development:** Writing a C++ pass to analyze and transform IR
* **Optimization:** Simplifying gate sequences (e.g., cancelling `X; X`)

## Features

* Custom MLIR dialect (**QuantumDialect**) with basic operations:
  * `quantum.x` - Pauli-X gate (bit flip)
  * `quantum.h` - Hadamard gate (superposition)
  * `quantum.cx` - Controlled-X gate (CNOT)
  * `quantum.alloc` - Allocate quantum qubits
* C++ pass that detects consecutive `quantum.x` gates on the same qubit and removes them
* Comprehensive test suite with example MLIR files
* Python bindings for programmatic circuit generation

## Example

**Input MLIR:**
```mlir
module {
  func.func @double_x_test() {
    %q = "quantum.alloc"() : () -> i32
    "quantum.x"(%q) : (i32) -> ()
    "quantum.x"(%q) : (i32) -> ()
    "quantum.h"(%q) : (i32) -> ()
    return
  }
}
```

**After optimization:**
```mlir
module {
  func.func @double_x_test() {
    %q = "quantum.alloc"() : () -> i32
    "quantum.h"(%q) : (i32) -> ()
    return
  }
}
```

## How to Use These Sources

This repository is designed to be **integrated into an LLVM/MLIR source tree**.

### 1. Clone LLVM Project

```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
```

### 2. Copy This Repository

This repository contains **only** the dialect and pass source files, intended for integration into LLVM/MLIR.

```bash
# Copy the quantum dialect files to the MLIR source tree
cp -rp /path/to/this/repo/mlir/* ./mlir/
```

### 3. Update CMake Build Configuration

Add the Quantum dialect to MLIR's build system by updating the following files:

**In `mlir/include/mlir/Dialect/CMakeLists.txt`:**
```cmake
# Add this line in alphabetical order
add_subdirectory(Quantum)
```

**In `mlir/lib/Dialect/CMakeLists.txt`:**
```cmake
# Add this line in alphabetical order
add_subdirectory(Quantum)
```

**In `mlir/test/lib/Dialect/CMakeLists.txt`:**
```cmake
# Add this line in alphabetical order
add_subdirectory(Quantum)
```

### 4. Build LLVM/MLIR

```bash
mkdir build
cd build
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON
ninja
```

For detailed MLIR setup instructions, see [MLIR's official documentation](https://mlir.llvm.org/docs/).

### 5. Test the Integration

After building, you can test the quantum dialect using the provided test program:

```bash
# Run the quantum dialect test program
./build/bin/test-quantum-dialect mlir/test/Dialect/Quantum/double_x_test.mlir

# Or pipe MLIR content
cat mlir/test/Dialect/Quantum/double_x_test.mlir | ./build/bin/test-quantum-dialect
```

Expected output:
```
=== Original MLIR ===
module {
  func.func @double_x_test() {
    %0 = "quantum.alloc"() : () -> i32
    "quantum.x"(%0) : (i32) -> ()
    "quantum.x"(%0) : (i32) -> ()
    "quantum.h"(%0) : (i32) -> ()
    return
  }
}

=== Optimized MLIR ===
module {
  func.func @double_x_test() {
    %0 = "quantum.alloc"() : () -> i32
    "quantum.h"(%0) : (i32) -> ()
    return
  }
}
```

## Repository Structure

```
mlir/
├── include/mlir/Dialect/Quantum/        # Quantum dialect headers
│   ├── IR/                              # Operation and dialect definitions
│   │   ├── Quantum.h                    # Main dialect header
│   │   ├── QuantumOps.td               # TableGen operation definitions
│   │   └── CMakeLists.txt
│   ├── Passes/                          # Pass headers and definitions
│   │   ├── QuantumPasses.h             # Pass registration
│   │   ├── Passes.td                   # TableGen pass definitions
│   │   └── CMakeLists.txt
│   └── CMakeLists.txt
├── lib/Dialect/Quantum/                 # Quantum dialect implementation
│   ├── QuantumDialect.cpp              # Dialect implementation
│   ├── QuantumOps.cpp                  # Operation implementations
│   ├── Passes/                          # Pass implementations
│   │   └── QuantumCancelXPass.cpp      # X gate cancellation pass
│   └── CMakeLists.txt
├── test/Dialect/Quantum/                # MLIR test files
│   ├── bell_state.mlir                 # Bell state example
│   ├── double_x_test.mlir              # Double X cancellation test
│   ├── test-quantum-generic.mlir       # Generic quantum operations
│   └── lib/                            # Test programs
│       └── test-quantum.cpp            # Standalone test program
├── python/mlir/dialects/quantum/        # Python bindings
│   ├── __init__.py                     # Module exports
│   ├── emit_quantum_mlir.py            # Circuit to MLIR conversion
│   └── README.md                       # Python bindings documentation
└── test/python/dialects/quantum/        # Python tests
    └── test_quantum_dialect.py         # Python binding tests
```

## Python Bindings (Optional)

If you built with Python bindings enabled (`-DMLIR_ENABLE_BINDINGS_PYTHON=ON`), you can use the Python API:

```bash
# Run Python examples
python mlir/python/mlir/dialects/quantum/emit_quantum_mlir.py

# Generate and optimize quantum circuits
python mlir/python/mlir/dialects/quantum/emit_quantum_mlir.py | ./build/bin/test-quantum-dialect
```

## Additional Examples

The repository includes several example MLIR files:

- `mlir/test/Dialect/Quantum/bell_state.mlir` - Demonstrates creating a Bell state
- `mlir/test/Dialect/Quantum/double_x_test.mlir` - Tests X gate cancellation
- `mlir/test/Dialect/Quantum/test-quantum-generic.mlir` - Generic quantum operations

## Performance Considerations

The quantum dialect and optimization passes are designed to:
- **Preserve control flow integrity** - No modifications to function structure
- **Maintain debug information** - All operations retain location information
- **Avoid performance profile corruption** - No impact on profiling or instrumentation

## Requirements

- LLVM/MLIR source tree
- CMake 3.20 or later
- Ninja build system (recommended)
- C++17 compatible compiler
- Python 3.7+ (for Python bindings)

## License

This project follows the LLVM Project's licensing terms. See the main LLVM LICENSE.TXT for details.
