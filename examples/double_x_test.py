"""Double X gate test - demonstrates gate cancellation."""

from qmlir import Circuit


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
    print(f"Gates: {circuit.gates}")
    print()
    print("MLIR representation:")
    print(circuit.to_mlir())
    print()
    print("To test gate cancellation:")
    print("python examples/double_x_test.py | build/mlir/tools/quantum-opt --quantum-cancel-x")
    print()
    print("Expected result: The two X gates should be removed, leaving only H gates.")


if __name__ == "__main__":
    main()
