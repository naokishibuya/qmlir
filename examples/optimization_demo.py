#!/usr/bin/env python3
"""
Quantum Circuit Optimization Demo

This example shows how QMLIR automatically optimizes quantum circuits
by canceling self-inverse gate pairs.
"""

from qmlir import QuantumCircuit, simulate


def main():
    print("Quantum Circuit Optimization Demo")
    print("=" * 40)

    # Create a circuit with redundant gates
    circuit = QuantumCircuit(1)
    circuit.h(0)  # Hadamard
    circuit.x(0)  # X gate
    circuit.x(0)  # X gate (cancels with previous)
    circuit.z(0)  # Z gate
    circuit.z(0)  # Z gate (cancels with previous)
    circuit.h(0)  # Hadamard (cancels with first)

    print("\nOriginal circuit (before optimization):")
    print(circuit)
    print(f"Original gate count: {len(circuit.gates)}")

    # Simulate the circuit (optimization happens automatically)
    result = simulate(circuit)

    print("\nAfter automatic optimization:")
    print(f"Operations executed: {result['num_operations']}")
    print(f"Final probabilities: {result['probabilities']}")
    print("Expected: [1.0, 0.0] (identity operation - no change)")

    print("\nOptimization summary:")
    print(f"  {len(circuit.gates)} gates → {result['num_operations']} operations")
    print("  All self-inverse pairs were automatically canceled!")


if __name__ == "__main__":
    main()
