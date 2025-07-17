# QMLIR Architecture: Backend Support

## Current Structure (Single Backend)

```
qmlir/
├── ast.py          # Frontend: Circuit representation
├── codegen.py      # MLIR Backend: AST → MLIR
├── config.py       # MLIR tools integration
└── __init__.py     # Public API
```

## Future Structure (Multiple Backends)

```
qmlir/
├── ast.py              # Frontend: Circuit representation (backend-agnostic)
├── transpiler.py       # High-level transpiler interface
├── backends/
│   ├── __init__.py
│   ├── base.py         # Abstract backend interface
│   ├── mlir/
│   │   ├── __init__.py
│   │   ├── codegen.py  # MLIR code generation
│   │   ├── optimizer.py # MLIR optimization passes
│   │   └── config.py   # MLIR-specific configuration
│   ├── qiskit/
│   │   ├── __init__.py
│   │   ├── codegen.py  # Qiskit circuit generation
│   │   └── config.py   # Qiskit backend configuration
│   ├── cirq/
│   │   ├── __init__.py
│   │   ├── codegen.py  # Cirq circuit generation
│   │   └── config.py   # Cirq backend configuration
│   └── openqasm/
│       ├── __init__.py
│       ├── codegen.py  # OpenQASM code generation
│       └── config.py   # OpenQASM configuration
└── __init__.py         # Public API with backend selection
```

## Backend Interface Design

### Abstract Backend Class

```python
# qmlir/backends/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from ..ast import QuantumCircuit

class Backend(ABC):
    """Abstract base class for quantum backends."""
    
    @abstractmethod
    def name(self) -> str:
        """Return backend name."""
        pass
    
    @abstractmethod
    def compile(self, circuit: QuantumCircuit, **kwargs) -> Any:
        """Compile circuit to backend-specific representation."""
        pass
    
    @abstractmethod
    def optimize(self, compiled_circuit: Any, optimization_level: int = 1) -> Any:
        """Apply backend-specific optimizations."""
        pass
    
    @abstractmethod
    def execute(self, compiled_circuit: Any, **kwargs) -> Any:
        """Execute circuit on backend (if applicable)."""
        pass
    
    @abstractmethod
    def supported_gates(self) -> set:
        """Return set of supported gate names."""
        pass
```

### MLIR Backend Implementation

```python
# qmlir/backends/mlir/codegen.py
from ..base import Backend
from ...ast import QuantumCircuit
from .optimizer import run_quantum_optimizer

class MLIRBackend(Backend):
    """MLIR quantum dialect backend."""
    
    def name(self) -> str:
        return "mlir"
    
    def compile(self, circuit: QuantumCircuit, function_name: str = "main") -> str:
        """Compile to MLIR representation."""
        # Current circuit_to_mlir logic
        return self._circuit_to_mlir(circuit, function_name)
    
    def optimize(self, mlir_code: str, optimization_level: int = 1) -> str:
        """Apply MLIR optimization passes."""
        if optimization_level == 0:
            return mlir_code
        elif optimization_level >= 1:
            return run_quantum_optimizer(mlir_code, "--quantum-cancel-self-inverse")
    
    def execute(self, mlir_code: str, **kwargs):
        """MLIR doesn't execute - returns optimized code."""
        return mlir_code
    
    def supported_gates(self) -> set:
        return {"i", "x", "y", "z", "h", "cx"}
```

### Qiskit Backend Implementation

```python
# qmlir/backends/qiskit/codegen.py
from ..base import Backend
from ...ast import QuantumCircuit
try:
    from qiskit import QuantumCircuit as QiskitCircuit
    from qiskit.compiler import transpile as qiskit_transpile
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

class QiskitBackend(Backend):
    """Qiskit backend for quantum circuits."""
    
    def __init__(self):
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit not available. Install with: pip install qiskit")
    
    def name(self) -> str:
        return "qiskit"
    
    def compile(self, circuit: QuantumCircuit, **kwargs) -> QiskitCircuit:
        """Convert to Qiskit circuit."""
        qc = QiskitCircuit(circuit.num_qubits)
        
        for gate in circuit.gates:
            if gate.name == "h":
                qc.h(gate.q[0])
            elif gate.name == "x":
                qc.x(gate.q[0])
            elif gate.name == "cx":
                qc.cx(gate.q[0], gate.q[1])
            # ... other gates
        
        return qc
    
    def optimize(self, qiskit_circuit: QiskitCircuit, optimization_level: int = 1):
        """Apply Qiskit optimization."""
        return qiskit_transpile(qiskit_circuit, optimization_level=optimization_level)
    
    def execute(self, qiskit_circuit: QiskitCircuit, backend_name: str = "qasm_simulator"):
        """Execute on Qiskit backend."""
        from qiskit import Aer, execute
        backend = Aer.get_backend(backend_name)
        return execute(qiskit_circuit, backend)
    
    def supported_gates(self) -> set:
        return {"i", "x", "y", "z", "h", "cx", "rx", "ry", "rz", "ccx"}
```

## Unified Transpiler Interface

```python
# qmlir/transpiler.py
from typing import Any, Dict, Optional, Union
from .ast import QuantumCircuit
from .backends.base import Backend
from .backends.mlir.codegen import MLIRBackend
from .backends.qiskit.codegen import QiskitBackend

AVAILABLE_BACKENDS = {
    "mlir": MLIRBackend,
    "qiskit": QiskitBackend,
}

def transpile(
    circuit: QuantumCircuit,
    backend: Union[str, Backend] = "mlir",
    optimization_level: int = 1,
    **kwargs
) -> Any:
    """Transpile quantum circuit to target backend.
    
    Args:
        circuit: QuantumCircuit to transpile
        backend: Target backend name or Backend instance
        optimization_level: Optimization level (0=none, 1=basic, 2=aggressive)
        **kwargs: Backend-specific arguments
    
    Returns:
        Backend-specific compiled representation
    """
    # Get backend instance
    if isinstance(backend, str):
        if backend not in AVAILABLE_BACKENDS:
            raise ValueError(f"Unknown backend: {backend}")
        backend_instance = AVAILABLE_BACKENDS[backend]()
    else:
        backend_instance = backend
    
    # Validate gates are supported
    circuit_gates = {gate.name for gate in circuit.gates}
    supported_gates = backend_instance.supported_gates()
    unsupported = circuit_gates - supported_gates
    if unsupported:
        raise ValueError(f"Backend {backend_instance.name()} doesn't support gates: {unsupported}")
    
    # Compile and optimize
    compiled = backend_instance.compile(circuit, **kwargs)
    if optimization_level > 0:
        compiled = backend_instance.optimize(compiled, optimization_level)
    
    return compiled
```

## Usage Examples

### Multi-Backend Usage

```python
from qmlir import QuantumCircuit, transpile

# Create circuit once
circuit = QuantumCircuit(2)
circuit.h(0).cx(0, 1)

# Transpile to different backends
mlir_code = transpile(circuit, backend="mlir", optimization_level=1)
qiskit_circuit = transpile(circuit, backend="qiskit", optimization_level=1)

# Execute on different backends
from qmlir.backends.qiskit import QiskitBackend
qiskit_backend = QiskitBackend()
result = qiskit_backend.execute(qiskit_circuit, backend_name="qasm_simulator")
```

### Backend-Specific Features

```python
# MLIR backend with custom passes
mlir_result = transpile(
    circuit, 
    backend="mlir",
    optimization_level=2,
    passes=["--quantum-cancel-self-inverse", "--custom-pass"]
)

# Qiskit backend with hardware constraints
qiskit_result = transpile(
    circuit,
    backend="qiskit",
    optimization_level=1,
    coupling_map=[[0, 1], [1, 2]],
    basis_gates=["u1", "u2", "u3", "cx"]
)
```

## Benefits of This Architecture

1. **Separation of Concerns**: Frontend (AST) independent of backends
2. **Extensibility**: Easy to add new backends without changing core API
3. **Consistency**: Uniform interface across all backends
4. **Optional Dependencies**: Backends can have optional dependencies
5. **Backend-Specific Features**: Each backend can expose its unique capabilities
6. **Testing**: Each backend can be tested independently

## Migration Path

1. **Phase 1**: Move current MLIR code to `backends/mlir/` (minimal changes)
2. **Phase 2**: Create abstract backend interface
3. **Phase 3**: Add second backend (Qiskit) to validate interface
4. **Phase 4**: Add more backends (Cirq, OpenQASM, etc.)
5. **Phase 5**: Add backend-specific optimizations and features

This architecture naturally accommodates your vision of backend support while maintaining the current API and functionality.
