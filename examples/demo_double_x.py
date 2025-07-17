"""Simple demonstration of double X gate cancellation."""

import sys
from qmlir import Circuit, circuit_to_mlir


def main():
    """Demonstrate double X gate cancellation."""
    # Create a circuit with two X gates on the same qubit
    circuit = Circuit()
    circuit.x(0).x(0)

    # Generate MLIR
    mlir_code = circuit_to_mlir(circuit, "double_x_test")

    # Check if output is being piped
    if sys.stdout.isatty():
        # Interactive mode - show description
        print("=== Double X Gate Test Demo ===")
        print(f"Circuit: {circuit}")
        print()
        print("Generated MLIR:")
        print(mlir_code)
        print()
        print("To test X gate cancellation:")
        print("python examples/demo_double_x.py | quantum-opt --quantum-cancel-x")
    else:
        # Piped mode - only output MLIR
        print(mlir_code)


if __name__ == "__main__":
    main()
