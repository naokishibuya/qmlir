"""Simple demonstration of Bell state circuit generation."""

import sys

from qmlir import Circuit, circuit_to_mlir


def main():
    """Demonstrate Bell state circuit generation."""
    # Create a Bell state circuit
    circuit = Circuit()
    circuit.h(0).cx(0, 1)

    # Generate MLIR
    mlir_code = circuit_to_mlir(circuit, "bell_state")

    # Check if output is being piped
    if sys.stdout.isatty():
        # Interactive mode - show description
        print("=== Bell State Circuit Demo ===")
        print(f"Circuit: {circuit}")
        print()
        print("Generated MLIR:")
        print(mlir_code)
        print()
        print("To test with quantum-opt:")
        print("python examples/demo_bell_state.py | quantum-opt")
    else:
        # Piped mode - only output MLIR
        print(mlir_code)


if __name__ == "__main__":
    main()
