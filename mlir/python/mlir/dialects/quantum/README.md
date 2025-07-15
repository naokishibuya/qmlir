# Quantum Dialect Python Bindings

This directory contains Python bindings and utilities for the MLIR Quantum dialect.

## Structure

```
mlir/python/mlir/dialects/quantum/
├── __init__.py                    # Main module exports
├── emit_quantum_mlir.py          # Circuit to MLIR conversion + examples
├── quantum_bindings.py           # (Future) Generated bindings
└── README.md                     # This file

mlir/test/python/dialects/quantum/
├── test_quantum_dialect.py      # Python binding tests
└── test_emit_quantum.py         # Circuit generation tests
```

## Usage

### Basic Circuit Creation

```python
from mlir.dialects.quantum import Circuit, circuit_to_mlir

# Create a simple circuit
circ = Circuit()
circ.x(0)      # X gate on qubit 0
circ.x(0)      # Another X gate (will be optimized away)
circ.h(0)      # Hadamard gate

# Convert to MLIR
module = circuit_to_mlir(circ)
print(module)
```

### Bell State Example

```python
# Create a Bell state |00⟩ + |11⟩
circ = Circuit()
circ.h(0)      # Put qubit 0 in superposition
circ.cx(0, 1)  # Entangle with qubit 1

module = circuit_to_mlir(circ)
print(module)
```

### Integration with Optimization

```python
# Generate MLIR and pipe to optimization
import subprocess

circ = Circuit()
circ.x(0)
circ.x(0)  # Redundant - will be removed
circ.h(0)

module = circuit_to_mlir(circ)
result = subprocess.run(
    ["./build/bin/test-quantum-dialect"],
    input=str(module),
    text=True,
    capture_output=True
)
print("Optimized MLIR:")
print(result.stdout)
```

## Requirements

- MLIR built with Python bindings enabled (`-DMLIR_ENABLE_BINDINGS_PYTHON=ON`)
- Quantum dialect built and available in the library path
- Python 3.7+

## Running Examples

```bash
# Run the built-in examples (from the project root)
python mlir/python/mlir/dialects/quantum/emit_quantum_mlir.py

# Run with optimization
python mlir/python/mlir/dialects/quantum/emit_quantum_mlir.py | ./build/bin/test-quantum-dialect

# Run tests
python mlir/test/python/dialects/quantum/test_quantum_dialect.py

# Alternative: from the quantum dialect directory
cd mlir/python/mlir/dialects/quantum
python emit_quantum_mlir.py
```

## Library Loading

The quantum dialect library needs to be loaded for the Python bindings to work. The scripts automatically try to load:

- `libMLIRQuantum.dylib` (macOS)
- `libMLIRQuantum.so` (Linux)
- `MLIRQuantum.dll` (Windows)

From the following locations:
1. System library path
2. `build/lib/` directory
3. `lib/` directory

## Future Enhancements

- Auto-generated Python bindings using MLIR's Python binding generator
- Type-safe quantum type system
- Circuit optimization passes accessible from Python
- Quantum simulator integration
- Visualization tools
