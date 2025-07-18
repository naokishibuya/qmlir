"""Example demonstrating Phase 2 parametric circuits with MLIR backend support.

This example shows how to use parametric quantum circuits with the QMLIR compiler,
including MLIR dialect support for parametric gates and optimization passes.
"""

from qmlir import QuantumCircuit, Parameter, transpile, circuit_to_mlir


def vqe_ansatz_example():
    """Example: VQE ansatz circuit with parametric gates."""
    print("=== VQE Ansatz Example ===")

    # Create a simple VQE ansatz for H2 molecule
    circuit = QuantumCircuit(2)

    # Define parameters
    theta1 = Parameter(0.1, "theta1")  # First rotation angle
    theta2 = Parameter(0.2, "theta2")  # Second rotation angle

    # Build ansatz
    circuit.ry(0, theta1)  # Prepare qubit 0
    circuit.ry(1, theta2)  # Prepare qubit 1
    circuit.cx(0, 1)  # Entangle qubits
    circuit.ry(0, theta1)  # Apply rotation again (parameter reuse)

    print("Circuit gates:")
    for i, gate in enumerate(circuit.gates):
        params_str = f", params={gate.parameters}" if gate.parameters else ""
        print(f"  {i}: {gate.name}({gate.q}{params_str})")

    # Generate MLIR
    mlir_code = circuit_to_mlir(circuit, "vqe_ansatz")
    print(f"\nGenerated MLIR:\n{mlir_code}")

    # Transpile with optimization
    optimized = transpile(circuit, optimization_level=1)
    print(f"Optimized MLIR:\n{optimized}")

    return circuit


def parametric_optimization_example():
    """Example: Optimization behavior with parametric gates."""
    print("\n=== Parametric Optimization Example ===")

    circuit = QuantumCircuit(2)
    theta = Parameter(0.5, "theta")

    # Mix parametric and fixed gates
    circuit.h(0)
    circuit.h(0)  # These H gates should cancel
    circuit.rx(1, theta)
    circuit.rx(1, theta)  # These RX gates should NOT cancel (they combine)
    circuit.x(0)
    circuit.x(0)  # These X gates should cancel
    circuit.cx(0, 1)
    circuit.y(1)
    circuit.y(1)  # These Y gates should cancel

    print("Original circuit:")
    for gate in circuit.gates:
        params_str = f"({gate.parameters[0].name})" if gate.parameters else ""
        print(f"  {gate.name}{params_str} on {gate.q}")

    # Compare unoptimized vs optimized
    unoptimized = transpile(circuit, optimization_level=0)
    optimized = transpile(circuit, optimization_level=1)

    print("\nUnoptimized gate counts:")
    print(f"  H gates: {unoptimized.count('quantum.h')}")
    print(f"  RX gates: {unoptimized.count('quantum.rx')}")
    print(f"  X gates: {unoptimized.count('quantum.x')}")
    print(f"  Y gates: {unoptimized.count('quantum.y')}")
    print(f"  CX gates: {unoptimized.count('quantum.cx')}")

    print("\nOptimized gate counts:")
    print(f"  H gates: {optimized.count('quantum.h')} (should be 0 - cancelled)")
    print(f"  RX gates: {optimized.count('quantum.rx')} (should be 2 - not cancelled)")
    print(f"  X gates: {optimized.count('quantum.x')} (should be 0 - cancelled)")
    print(f"  Y gates: {optimized.count('quantum.y')} (should be 0 - cancelled)")
    print(f"  CX gates: {optimized.count('quantum.cx')} (should be 1 - unchanged)")

    print(f"\nOptimized MLIR:\n{optimized}")


def multi_parameter_example():
    """Example: Circuit with multiple independent parameters."""
    print("\n=== Multi-Parameter Example ===")

    circuit = QuantumCircuit(3)

    # Create multiple parameters
    alpha = Parameter(0.1, "alpha")
    beta = Parameter(0.2, "beta")
    gamma = Parameter(0.3, "gamma")

    # Use parameters in different gates
    circuit.rx(0, alpha)
    circuit.ry(1, beta)
    circuit.rz(2, gamma)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.rx(0, alpha)  # Reuse alpha
    circuit.ry(1, beta)  # Reuse beta

    print("Circuit with multiple parameters:")
    for gate in circuit.gates:
        if gate.parameters:
            param_name = gate.parameters[0].name
            print(f"  {gate.name}({gate.q[0]}, {param_name})")
        else:
            print(f"  {gate.name}({', '.join(map(str, gate.q))})")

    # Generate MLIR
    mlir_code = circuit_to_mlir(circuit, "multi_param")
    print(f"\nGenerated MLIR:\n{mlir_code}")

    # Verify parameter mapping
    print("Parameter mapping:")
    print("  alpha (used 2 times) -> %arg0")
    print("  beta (used 2 times) -> %arg1")
    print("  gamma (used 1 time) -> %arg2")


def layered_ansatz_example():
    """Example: Layered ansatz typical in variational algorithms."""
    print("\n=== Layered Ansatz Example ===")

    n_qubits = 4
    n_layers = 2
    circuit = QuantumCircuit(n_qubits)

    parameters = []

    # Build layered ansatz
    for layer in range(n_layers):
        print(f"\nLayer {layer}:")

        # Single-qubit rotations
        for qubit in range(n_qubits):
            param = Parameter(0.1 * (layer * n_qubits + qubit), f"theta_{layer}_{qubit}")
            parameters.append(param)
            circuit.ry(qubit, param)
            print(f"  RY({qubit}, {param.name})")

        # Entangling gates
        for qubit in range(n_qubits - 1):
            circuit.cx(qubit, qubit + 1)
            print(f"  CX({qubit}, {qubit + 1})")

    print(f"\nTotal parameters: {len(parameters)}")
    print(f"Total gates: {len(circuit.gates)}")

    # Generate MLIR
    mlir_code = circuit_to_mlir(circuit, "layered_ansatz")
    print("\nGenerated MLIR function signature:")
    lines = mlir_code.split("\n")
    for line in lines:
        if "@layered_ansatz" in line:
            print(f"  {line.strip()}")
            break

    # Count gates in MLIR
    ry_count = mlir_code.count('"quantum.ry"')
    cx_count = mlir_code.count('"quantum.cx"')
    print("\nMLIR gate counts:")
    print(f"  RY gates: {ry_count}")
    print(f"  CX gates: {cx_count}")

    return circuit


def main():
    """Run all examples."""
    print("Parametric Circuits Examples")
    print("=" * 50)

    # Run examples
    vqe_ansatz_example()
    parametric_optimization_example()
    multi_parameter_example()
    layered_ansatz_example()


if __name__ == "__main__":
    main()
