#!/usr/bin/env python3
"""
Bell State Example - Creating maximally entangled two-qubit states

This example demonstrates the most fundamental quantum entanglement
using QMLIR's simple API.
"""

from qmlir import QuantumCircuit, simulate


def main():
    print("Bell State Creation with QMLIR")
    print("=" * 35)

    # Create a Bell state: (|00⟩ + |11⟩)/√2
    circuit = QuantumCircuit(2)
    circuit.h(0)  # Put qubit 0 in superposition
    circuit.cx(0, 1)  # Entangle qubits 0 and 1

    print("\nCircuit:")
    print(circuit)

    # Simulate the circuit
    result = simulate(circuit)

    print("\nResults:")
    print(f"Probabilities: {result['probabilities']}")
    print("Expected: [0.5, 0.0, 0.0, 0.5]")
    print("Meaning: 50% chance of |00⟩, 50% chance of |11⟩")
    print(f"Operations executed: {result['num_operations']}")


if __name__ == "__main__":
    main()
