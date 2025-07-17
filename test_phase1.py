"""Test the parametric circuits Phase 1 implementation."""

from qmlir import QuantumCircuit, Parameter


def test_parameter_creation():
    """Test Parameter class creation and properties."""
    # Test with value and name
    theta = Parameter(1.57, "theta")
    assert theta.name == "theta"
    assert theta.initial_value == 1.57
    assert len(theta.id) == 36  # UUID length

    # Test with value only (auto-generated name)
    phi = Parameter(3.14159)
    assert phi.name.startswith("param_")
    assert phi.initial_value == 3.14159
    assert len(phi.id) == 36

    # Test with zero value and name
    omega = Parameter(0.0, "omega")
    assert omega.name == "omega"
    assert omega.initial_value == 0.0

    # Test equality (should be based on ID, not initial value)
    same_theta = theta
    assert theta == same_theta
    assert theta != phi

    # Test string representations
    assert "initial_value=1.57" in repr(theta)
    assert "theta=1.57" in str(theta)

    print("✓ Parameter creation tests passed")


def test_parametric_gates():
    """Test parametric gate creation."""
    circuit = QuantumCircuit(2)
    theta = Parameter(1.57, "theta")  # π/2
    phi = Parameter(0.785, "phi")  # π/4

    # Add parametric gates
    circuit.rx(0, theta)
    circuit.ry(1, phi)
    circuit.rz(0, theta)  # Reuse parameter

    # Verify gates were added
    assert len(circuit.gates) == 3

    # Check first gate
    rx_gate = circuit.gates[0]
    assert rx_gate.name == "rx"
    assert rx_gate.q == (0,)
    assert len(rx_gate.parameters) == 1
    assert rx_gate.parameters[0] == theta
    assert rx_gate.parameters[0].initial_value == 1.57

    # Check parameter reuse
    rz_gate = circuit.gates[2]
    assert rz_gate.parameters[0] == theta  # Same parameter object
    assert rz_gate.parameters[0].initial_value == 1.57

    print("✓ Parametric gate tests passed")


def test_parameter_collection():
    """Test parameter collection logic."""
    circuit = QuantumCircuit(2)
    theta = Parameter(1.57, "theta")
    phi = Parameter(0.785, "phi")

    # Create circuit with parameter reuse
    circuit.rx(0, theta)
    circuit.h(1)  # Non-parametric gate
    circuit.ry(1, phi)
    circuit.rz(0, theta)  # Reuse theta

    # Manually test parameter collection logic
    param_index_map = {}
    for gate in circuit.gates:
        for param in gate.parameters:
            if param.id not in param_index_map:
                param_index_map[param.id] = len(param_index_map)

    # Should have 2 unique parameters
    assert len(param_index_map) == 2
    assert theta.id in param_index_map
    assert phi.id in param_index_map

    # Check deterministic ordering
    assert param_index_map[theta.id] == 0  # First occurrence
    assert param_index_map[phi.id] == 1  # Second occurrence

    print("✓ Parameter collection tests passed")


def test_mixed_circuit():
    """Test circuit with both parametric and non-parametric gates."""
    circuit = QuantumCircuit(3)
    theta = Parameter(1.57, "rotation_angle")  # π/2

    # Build a mixed circuit
    circuit.h(0)
    circuit.rx(1, theta)
    circuit.cx(0, 1)
    circuit.ry(2, theta)  # Reuse parameter
    circuit.z(2)

    assert len(circuit.gates) == 5

    # Check that only rx and ry gates have parameters
    param_gates = [g for g in circuit.gates if g.parameters]
    non_param_gates = [g for g in circuit.gates if not g.parameters]

    assert len(param_gates) == 2
    assert len(non_param_gates) == 3

    # Check parameter values
    assert param_gates[0].parameters[0].initial_value == 1.57
    assert param_gates[1].parameters[0].initial_value == 1.57

    print("✓ Mixed circuit tests passed")


def test_common_parameter_patterns():
    """Test common parameter usage patterns."""
    import math

    # Common rotation angles
    pi_2 = Parameter(math.pi / 2, "pi_2")
    pi_4 = Parameter(math.pi / 4, "pi_4")
    pi = Parameter(math.pi, "pi")

    circuit = QuantumCircuit(3)
    circuit.rx(0, pi_2)  # 90° rotation
    circuit.ry(1, pi_4)  # 45° rotation
    circuit.rz(2, pi)  # 180° rotation

    # Check values
    assert abs(circuit.gates[0].parameters[0].initial_value - math.pi / 2) < 1e-10
    assert abs(circuit.gates[1].parameters[0].initial_value - math.pi / 4) < 1e-10
    assert abs(circuit.gates[2].parameters[0].initial_value - math.pi) < 1e-10

    print("✓ Common parameter patterns tests passed")


if __name__ == "__main__":
    test_parameter_creation()
    test_parametric_gates()
    test_parameter_collection()
    test_mixed_circuit()
    test_common_parameter_patterns()
    print("\n🎉 All Phase 1 tests passed! Parametric circuits API with initial values is working.")
