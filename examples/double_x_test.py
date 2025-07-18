"""Double X gate test - demonstrates gate cancellation."""

from qmlir import QuantumCircuit, circuit_to_mlir, optimize


def main():
    """Create a circuit with double X gates to test cancellation."""
    # Create a circuit with cancelling X gates
    circuit = QuantumCircuit(1)

    # Start with Hadamard
    circuit.h(0)

    # Add two X gates (should cancel)
    circuit.x(0)
    circuit.x(0)

    # Add another Hadamard
    circuit.h(0)

    print("Double X Gate Test Circuit:")
    print(circuit.gates)
    print()

    # Generate MLIR
    mlir_code = circuit_to_mlir(circuit, "double_x_test")

    print("Original MLIR:")
    print(mlir_code)
    print()

    # Run optimization
    result = optimize(mlir_code, "--quantum-cancel-self-inverse")
    if result.returncode == 0:
        print("Optimized MLIR (self-inverse gates cancelled):")
        print(result.stdout.strip())
        print()
        print("Note: Both X-X and H-H gates are cancelled, leaving only qubit allocation!")
    else:
        print(f"Optimization failed: {result.stderr}")


if __name__ == "__main__":
    main()
