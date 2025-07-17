#!/usr/bin/env python3
"""Demo of the phase gates and controlled gates in QMLIR.

This demonstrates the S, T, S†, T†, CY, and CZ gates with MLIR generation.
Phase gates (S, T) are essential for Clifford+T universal quantum computation.
"""

from qmlir import QuantumCircuit, Parameter, transpile
from qmlir.compiler import circuit_to_mlir


def demo_phase_gates():
    """Demo the phase gates (S, T, S†, T†)."""
    print("=== Phase Gates Demo ===")

    circuit = QuantumCircuit(2)

    # Phase gates
    circuit.s(0)  # S gate: phase π/2
    circuit.t(0)  # T gate: phase π/4
    circuit.sdg(1)  # S† gate: phase -π/2
    circuit.tdg(1)  # T† gate: phase -π/4

    print("Gates added:")
    print("  S(0)   - Phase gate (π/2)")
    print("  T(0)   - T gate (π/4)")
    print("  S†(1)  - S-dagger gate (-π/2)")
    print("  T†(1)  - T-dagger gate (-π/4)")

    mlir_code = circuit_to_mlir(circuit, "phase_gates_demo")
    print(f"\nGenerated MLIR:\n{mlir_code}")

    return circuit


def demo_controlled_gates():
    """Demo the controlled gates (CY, CZ)."""
    print("\n=== Controlled Gates Demo ===")

    circuit = QuantumCircuit(3)

    # Controlled gates
    circuit.cy(0, 1)  # Controlled-Y
    circuit.cz(1, 2)  # Controlled-Z
    circuit.cx(0, 2)  # For comparison

    print("Gates added:")
    print("  CY(0,1) - Controlled-Y gate")
    print("  CZ(1,2) - Controlled-Z gate")
    print("  CX(0,2) - Controlled-X gate (for comparison)")

    mlir_code = circuit_to_mlir(circuit, "controlled_gates_demo")
    print(f"\nGenerated MLIR:\n{mlir_code}")

    return circuit


def demo_clifford_t_gates():
    """Demo Clifford+T gate set."""
    print("\n=== Clifford+T Gate Set Demo ===")

    circuit = QuantumCircuit(2)

    # Clifford gates
    circuit.h(0)  # Hadamard
    circuit.s(0)  # S gate
    circuit.cx(0, 1)  # CNOT

    # T gates (make it universal)
    circuit.t(0)  # T gate
    circuit.tdg(1)  # T-dagger

    print("Clifford+T gates:")
    print("  H(0)    - Hadamard (Clifford)")
    print("  S(0)    - S gate (Clifford)")
    print("  CX(0,1) - CNOT (Clifford)")
    print("  T(0)    - T gate (makes it universal)")
    print("  T†(1)   - T-dagger (makes it universal)")

    mlir_code = circuit_to_mlir(circuit, "clifford_t_demo")
    print(f"\nGenerated MLIR:\n{mlir_code}")

    return circuit


def demo_gate_relationships():
    """Demo relationships between gates."""
    print("\n=== Gate Relationships Demo ===")

    circuit = QuantumCircuit(2)

    # S and T gate relationships
    circuit.s(0)  # S = RZ(π/2)
    circuit.s(0)  # S² = Z
    circuit.t(1)  # T = RZ(π/4)
    circuit.t(1)  # T² = S

    print("Gate relationships:")
    print("  S = RZ(π/2)")
    print("  S² = Z (two S gates = Z gate)")
    print("  T = RZ(π/4)")
    print("  T² = S (two T gates = S gate)")
    print("  S† = RZ(-π/2) (inverse of S)")
    print("  T† = RZ(-π/4) (inverse of T)")

    mlir_code = circuit_to_mlir(circuit, "gate_relationships_demo")
    print(f"\nGenerated MLIR:\n{mlir_code}")

    return circuit


def demo_mixed_parametric_and_phase_gates():
    """Demo mixing parametric gates with phase gates."""
    print("\n=== Mixed Parametric and Phase Gates Demo ===")

    circuit = QuantumCircuit(2)
    theta = Parameter(0.5, "theta")

    # Mix parametric and fixed gates
    circuit.s(0)  # Fixed S gate
    circuit.ry(0, theta)  # Parametric RY gate
    circuit.t(0)  # Fixed T gate
    circuit.cy(0, 1)  # Fixed CY gate
    circuit.rz(1, theta)  # Parametric RZ gate (reuse parameter)
    circuit.cz(0, 1)  # Fixed CZ gate

    print("Mixed circuit:")
    print("  S(0)       - Fixed phase gate")
    print("  RY(0, θ)   - Parametric rotation")
    print("  T(0)       - Fixed T gate")
    print("  CY(0,1)    - Fixed controlled-Y")
    print("  RZ(1, θ)   - Parametric rotation (reuse θ)")
    print("  CZ(0,1)    - Fixed controlled-Z")

    mlir_code = circuit_to_mlir(circuit, "mixed_gates_demo")
    print(f"\nGenerated MLIR:\n{mlir_code}")

    return circuit


def demo_quantum_opt_integration():
    """Demo quantum-opt integration with new gates."""
    print("\n=== Quantum-Opt Integration Demo ===")

    circuit = QuantumCircuit(2)

    # Add gates that might be optimized
    circuit.s(0)
    circuit.sdg(0)  # S and S† should cancel
    circuit.t(0)
    circuit.tdg(0)  # T and T† should cancel
    circuit.cy(0, 1)
    circuit.h(0)
    circuit.h(0)  # H and H should cancel

    print("Before optimization:")
    print("  S(0), S†(0) - should cancel")
    print("  T(0), T†(0) - should cancel")
    print("  CY(0,1)     - should remain")
    print("  H(0), H(0)  - should cancel")

    unoptimized = transpile(circuit, optimization_level=0)
    optimized = transpile(circuit, optimization_level=1)

    print("\nUnoptimized gate counts:")
    print(f"  S: {unoptimized.count('quantum.s')}")
    print(f"  S†: {unoptimized.count('quantum.sdg')}")
    print(f"  T: {unoptimized.count('quantum.t')}")
    print(f"  T†: {unoptimized.count('quantum.tdg')}")
    print(f"  CY: {unoptimized.count('quantum.cy')}")
    print(f"  H: {unoptimized.count('quantum.h')}")

    print("\nOptimized gate counts:")
    print(f"  S: {optimized.count('quantum.s')}")
    print(f"  S†: {optimized.count('quantum.sdg')}")
    print(f"  T: {optimized.count('quantum.t')}")
    print(f"  T†: {optimized.count('quantum.tdg')}")
    print(f"  CY: {optimized.count('quantum.cy')}")
    print(f"  H: {optimized.count('quantum.h')}")

    print(f"\nOptimized MLIR:\n{optimized}")

    return circuit


def main():
    """Run all demos."""
    print("QMLIR Phase Gates and Controlled Gates Demo")
    print("=" * 50)

    demo_phase_gates()
    demo_controlled_gates()
    demo_clifford_t_gates()
    demo_gate_relationships()
    demo_mixed_parametric_and_phase_gates()
    demo_quantum_opt_integration()


if __name__ == "__main__":
    main()
