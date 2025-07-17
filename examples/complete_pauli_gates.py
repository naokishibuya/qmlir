"""
Complete Pauli Gate Set Example
Demonstrates all gates in the quantum compiler:
- I: Identity gate
- X: Pauli-X (NOT gate)
- Y: Pauli-Y gate
- Z: Pauli-Z gate
- H: Hadamard gate
- CX: Controlled-X (CNOT gate)
"""

from qmlir import QuantumCircuit, circuit_to_mlir
from qmlir.config import run_quantum_optimizer

# Create a circuit with all gates (specify number of qubits)
circuit = QuantumCircuit(3)

# Single-qubit gates
circuit.i(0)  # Identity
circuit.x(0)  # Pauli-X
circuit.y(0)  # Pauli-Y
circuit.z(0)  # Pauli-Z
circuit.h(0)  # Hadamard

# Two-qubit gates
circuit.cx(0, 1)  # CNOT
circuit.cx(1, 2)  # Another CNOT

# Add some cancellations to demonstrate optimization
circuit.h(1)  # H
circuit.h(1)  # H (should cancel with previous)

circuit.x(2)  # X
circuit.x(2)  # X (should cancel with previous)

circuit.cx(0, 2)  # CX
circuit.cx(0, 2)  # CX (should cancel with previous)

mlir_code = circuit_to_mlir(circuit, "complete_pauli_gates")

print("Complete Pauli Gate Set Example")
print("=" * 40)
print("Original circuit:")
print(circuit)

print("\nMLIR representation:")
print(mlir_code)

print("\nAfter optimization:")
optimized = run_quantum_optimizer(mlir_code, "--quantum-cancel-self-inverse")
print(optimized.stdout.strip())
