#!/usr/bin/env python3
"""
Parameterized Quantum Circuit Demo

This example demonstrates how to use Parameter objects for
variable quantum rotations.
"""

from qmlir import QuantumCircuit, Parameter, simulate
import math


def main():
    print("Parameterized Quantum Circuit Demo")
    print("=" * 40)

    # Create a parameterized rotation circuit
    angle = Parameter(math.pi / 2, "rotation_angle")  # 90 degrees

    circuit = QuantumCircuit(1)
    circuit.ry(0, angle)  # Rotate around Y-axis

    print("\nParameterized circuit:")
    print(circuit)

    # Simulate the circuit
    result = simulate(circuit)

    print("\nResults:")
    print(f"Probabilities: {result['probabilities']}")
    print("Expected: [0.5, 0.5] for RY(π/2)")
    print("This creates an equal superposition state!")

    # Try different angles
    print("\nExploring different rotation angles:")
    angles = [0, math.pi / 4, math.pi / 2, math.pi]
    names = ["0", "π/4", "π/2", "π"]

    for angle_val, name in zip(angles, names):
        param = Parameter(angle_val, f"theta_{name}")
        test_circuit = QuantumCircuit(1)
        test_circuit.ry(0, param)

        result = simulate(test_circuit)
        p0, p1 = result["probabilities"]
        print(f"  RY({name}): P(|0⟩)={p0:.3f}, P(|1⟩)={p1:.3f}")


if __name__ == "__main__":
    main()
