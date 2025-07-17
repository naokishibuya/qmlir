"""Double X gate test - demonstrates gate cancellation."""

import subprocess
from qmlir import Circuit, circuit_to_mlir
from qmlir.config import get_quantum_opt_path


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
    try:
        quantum_opt_path = get_quantum_opt_path()
        result = subprocess.run(
            [quantum_opt_path, "--quantum-cancel-x"], input=mlir_code, capture_output=True, text=True, timeout=10
        )

        if result.returncode == 0:
            print("Optimized MLIR (X gates cancelled):")
            print(result.stdout.strip())
        else:
            print(f"Optimization failed: {result.stderr}")
    except Exception as e:
        print(f"Error running optimization: {e}")
        print("To manually test optimization:")
        print("python examples/double_x_test.py | build/mlir/tools/quantum-opt --quantum-cancel-x")


if __name__ == "__main__":
    main()
