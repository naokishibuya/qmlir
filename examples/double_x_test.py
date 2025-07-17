"""Double X gate test - demonstrates gate cancellation."""

from qmlir import Circuit, circuit_to_mlir, run_quantum_opt


def main():
    """Create a circuit with double X gates to test cancellation."""
    # Create a circuit with cancelling X gates
    circuit = Circuit()

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
    result = run_quantum_opt(mlir_code, "--quantum-cancel-x")
    if result.returncode == 0:
        print("Optimized MLIR (X gates cancelled):")
        print(result.stdout.strip())
    else:
        print(f"Optimization failed: {result.stderr}")


if __name__ == "__main__":
    main()
