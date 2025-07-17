"""Self-inverse gate cancellation test - demonstrates involutory gate optimization."""

from qmlir import QuantumCircuit, transpile


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

    # Generate original MLIR (no optimization)
    original_mlir = transpile(circuit, optimization_level=0, function_name="cancel_self_inverse_test")

    print("Original MLIR:")
    print(original_mlir)
    print()

    # Generate optimized MLIR
    try:
        optimized_mlir = transpile(circuit, optimization_level=1, function_name="cancel_self_inverse_test")
        print("Optimized MLIR (self-inverse gates cancelled):")
        print(optimized_mlir.strip())
        print()
        print("Note: All self-inverse gates (X-X, Y-Y, Z-Z, H-H) are now cancelled!")
        print("The optimization can cancel identical gates on the same qubit even when")
        print("separated by operations on other qubits.")
        print("Only qubit allocations remain - the circuit is fully optimized.")
    except RuntimeError as e:
        print(f"Optimization failed: {e}")


if __name__ == "__main__":
    main()
