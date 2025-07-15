# MLIR Quantum Dialect Integration

A comprehensive quantum computing dialect for MLIR with optimization passes and Python bindings.

## Overview

This project provides a minimal but complete quantum computing dialect for MLIR, including:

- **Quantum Operations**: Basic quantum gates (alloc, x, h, cx)
- **Optimization Passes**: Gate cancellation and other quantum-specific optimizations
- **Python Bindings**: Programmatic circuit generation and MLIR emission
- **Test Suite**: Comprehensive tests and examples

## Project Structure

```
mlir/
├── include/mlir/Dialect/Quantum/        # Quantum dialect headers
│   ├── IR/                              # Operation and dialect definitions
│   ├── Passes/                          # Pass headers and TableGen definitions
│   └── README.md                        # → This file (central documentation)
├── lib/Dialect/Quantum/                 # Quantum dialect implementation
│   ├── Passes/                          # Pass implementations
│   └── CMakeLists.txt                   # Build configuration
├── test/Dialect/Quantum/                # MLIR test files
│   ├── *.mlir                           # Test cases and examples
│   └── lib/                             # Test programs
├── python/mlir/dialects/quantum/        # Python bindings
│   ├── emit_quantum_mlir.py            # Circuit to MLIR conversion
│   └── README.md                        # → Python bindings documentation
└── test/python/dialects/quantum/        # Python tests
```

## Quick Start

### 1. Build Integration

The Quantum dialect is integrated into MLIR's build system. Build MLIR with:

```bash
cmake -G Ninja ../llvm -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_TARGETS_TO_BUILD="host" -DCMAKE_BUILD_TYPE=Release
ninja
```

### 2. Test the Dialect

```bash
# Run the quantum dialect test program
./build/bin/test-quantum-dialect mlir/test/Dialect/Quantum/double_x_test.mlir

# Or pipe MLIR content
cat mlir/test/Dialect/Quantum/double_x_test.mlir | ./build/bin/test-quantum-dialect
```

### 3. Use Python Bindings

```bash
# Run Python examples (requires MLIR Python bindings)
python mlir/python/mlir/dialects/quantum/emit_quantum_mlir.py

# Generate and optimize quantum circuits
python mlir/python/mlir/dialects/quantum/emit_quantum_mlir.py | ./build/bin/test-quantum-dialect
```

## Documentation

### 📖 **Core Dialect Documentation**
- **Operations**: See `IR/QuantumOps.td` for quantum gate definitions
- **Passes**: See `Passes/Passes.td` for optimization pass definitions
- **Implementation**: See `lib/Dialect/Quantum/` for C++ implementation
- **Tests**: See `test/Dialect/Quantum/` for MLIR test cases

### 🐍 **[Python Bindings Documentation](../../../python/mlir/dialects/quantum/README.md)**
- Circuit creation and MLIR generation
- Python API reference and examples
- Integration with optimization passes
- Requirements and setup instructions

## Features

### Quantum Operations
- `quantum.alloc` - Allocate quantum qubits
- `quantum.x` - Pauli-X gate (bit flip)
- `quantum.h` - Hadamard gate (superposition)
- `quantum.cx` - Controlled-X gate (CNOT)

### Optimization Passes
- **X Gate Cancellation**: Removes consecutive X gates on the same qubit (X² = I)
- **Extensible Framework**: Easy to add new quantum-specific optimizations

### Python Integration
- **Programmatic Circuit Generation**: Create quantum circuits using Python
- **MLIR Emission**: Convert circuits to MLIR without string manipulation
- **Optimization Pipeline**: Seamlessly integrate with MLIR optimization passes

## Example Usage

### MLIR Quantum Circuit
```mlir
module {
  func.func @bell_state() {
    %q0 = "quantum.alloc"() : () -> i32
    %q1 = "quantum.alloc"() : () -> i32
    "quantum.h"(%q0) : (i32) -> ()
    "quantum.cx"(%q0, %q1) : (i32, i32) -> ()
    return
  }
}
```

### Python Circuit Generation
```python
from mlir.dialects.quantum import Circuit, circuit_to_mlir

# Create a Bell state circuit
circ = Circuit()
circ.h(0)      # Hadamard on qubit 0
circ.cx(0, 1)  # CNOT from qubit 0 to 1

# Convert to MLIR
module = circuit_to_mlir(circ)
print(module)
```

### Optimization Example
```bash
# Before optimization: X; X; H
# After optimization: H (consecutive X gates cancelled)
```

## Integration Status

- ✅ **Build System**: Fully integrated with MLIR's CMake build
- ✅ **Dialect Registration**: Automatic registration with MLIR context
- ✅ **Pass Framework**: Integrated with MLIR's pass manager
- ✅ **Test Suite**: Comprehensive test coverage
- ✅ **Python Bindings**: Full Python API support
- ✅ **Documentation**: Complete usage and API documentation

## Distribution

### For Core Dialect Usage
The quantum dialect is self-contained within the MLIR tree and requires no external dependencies beyond MLIR itself.

### For Python Bindings
The Python bindings and utilities are designed to be distributable as a separate package. See the [Python README](../../../python/mlir/dialects/quantum/README.md) for packaging and distribution details.

## Performance Considerations

The quantum dialect and its optimization passes are designed to:
- **Preserve control flow integrity** - No modifications to function structure
- **Maintain debug information** - All operations retain location information  
- **Avoid performance profile corruption** - No impact on profiling or instrumentation

## Future Enhancements

- Additional quantum gate operations (Y, Z, rotation gates)
- More sophisticated optimization passes (gate fusion, circuit depth reduction)
- Quantum type system with proper qubit types
- Integration with quantum simulators and hardware backends
- Advanced Python tooling (visualization, circuit analysis)

## Contributing

When extending the quantum dialect:

1. **Add new operations**: Update `QuantumOps.td` and implement in `QuantumOps.cpp`
2. **Add new passes**: Define in `Passes.td` and implement in `Passes/`
3. **Update tests**: Add test cases in `test/Dialect/Quantum/`
4. **Update Python bindings**: Extend the Python API as needed
5. **Update documentation**: Keep README files current

## License

This project follows the LLVM Project's licensing terms. See the main LLVM LICENSE.TXT for details.
