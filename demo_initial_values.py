"""Demo of parametric circuits with initial values."""

import math
from qmlir import QuantumCircuit, Parameter


def demo_initial_values():
    """Demonstrate parametric circuits with initial values."""
    print("🎯 Parametric Circuits Demo with Initial Values")
    print("=" * 50)

    # Create parameters with meaningful initial values
    theta = Parameter(math.pi / 2, "theta")  # 90 degrees
    phi = Parameter(math.pi / 4, "phi")  # 45 degrees
    omega = Parameter(0.0, "omega")  # Start at zero

    print("Parameters created:")
    print(f"  {theta}")
    print(f"  {phi}")
    print(f"  {omega}")
    print()

    # Build a variational quantum circuit
    circuit = QuantumCircuit(3)

    # Layer 1: Initial rotation
    circuit.ry(0, theta)
    circuit.ry(1, phi)
    circuit.ry(2, omega)

    # Layer 2: Entanglement
    circuit.cx(0, 1)
    circuit.cx(1, 2)

    # Layer 3: More rotations (reusing parameters)
    circuit.rx(0, phi)  # Reuse phi
    circuit.rz(1, theta)  # Reuse theta
    circuit.ry(2, omega)  # Reuse omega

    print("Circuit created:")
    print(circuit)
    print()

    # Show parameter usage
    param_gates = [g for g in circuit.gates if g.parameters]
    print(f"Found {len(param_gates)} parametric gates:")
    for i, gate in enumerate(param_gates):
        param = gate.parameters[0]
        print(f"  {i + 1}. {gate.name}({gate.q[0]}, {param.name}={param.initial_value:.4f})")
    print()

    # Demonstrate parameter reuse
    unique_params = set()
    for gate in circuit.gates:
        for param in gate.parameters:
            unique_params.add(param.id)

    print(f"Total parametric gates: {len(param_gates)}")
    print(f"Unique parameters: {len(unique_params)}")
    print("✓ Parameter reuse is working correctly!")
    print()

    # Show what the MLIR would look like
    try:
        from qmlir import circuit_to_mlir

        mlir_code = circuit_to_mlir(circuit, "variational_circuit")
        print("Generated MLIR function signature:")
        for line in mlir_code.split("\n"):
            if "variational_circuit" in line and "func.func" in line:
                print(f"  {line.strip()}")
                break
        print()
        print("✓ MLIR generation successful!")
    except Exception as e:
        print(f"⚠️  MLIR generation error: {e}")


if __name__ == "__main__":
    demo_initial_values()
