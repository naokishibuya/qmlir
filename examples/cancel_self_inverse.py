"""Self-inverse gate cancellation test - demonstrates involutory gate optimization."""

from qmlir import QuantumCircuit, circuit_to_mlir, run_quantum_optimizer


def main():
    """Create a circuit with self-inverse gates to test cancellation."""
    # Create a circuit with cancelling self-inverse gates (specify number of qubits)
    circuit = QuantumCircuit(3)

    # Start with Hadamard
    circuit.h(0)

    # Add Pauli gates (all self-inverse)
    circuit.x(0).x(0)  # Should cancel
    circuit.y(1).y(1)  # Should cancel
    circuit.z(2).z(2)  # Should cancel

    # Add another Hadamard (should cancel with first one)
    circuit.h(0)

    print("Self-Inverse Gate Cancellation Test:")
    print(circuit.gates)
    print()

    # Generate MLIR
    mlir_code = circuit_to_mlir(circuit, "cancel_self_inverse_test")

    print("Original MLIR:")
    print(mlir_code)
    print()

    # Run optimization
    result = run_quantum_optimizer(mlir_code, "--quantum-cancel-self-inverse")
    if result.returncode == 0:
        print("Optimized MLIR (self-inverse gates cancelled):")
        print(result.stdout.strip())
        print()
        print("Note: All self-inverse gates (X-X, Y-Y, Z-Z, H-H) are cancelled!")
        print("Only qubit allocations and non-cancelled gates remain - the circuit is fully optimized.")
    else:
        print(f"Optimization failed: {result.stderr}")


if __name__ == "__main__":
    main()
